package runner

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/greynewell/mist-go/parallel"

	"github.com/greynewell/swe-bench-fast/internal/dataset"
	"github.com/greynewell/swe-bench-fast/internal/docker"
	"github.com/greynewell/swe-bench-fast/internal/eval"
	"github.com/greynewell/swe-bench-fast/internal/grading"
	"github.com/greynewell/swe-bench-fast/internal/parser"
	"github.com/greynewell/swe-bench-fast/internal/spec"
)

// Config holds all configuration for a run.
type Config struct {
	DatasetPath     string
	PredictionsPath string
	Workers         int
	Timeout         time.Duration
	RunID           string
	Arch            string
	ARM64Registry   string // pull ARM64-native images from here (e.g. "docker.io/greynewell/swe-bench-arm64")
	X86Registry     string // pull x86 images from here (e.g. "ghcr.io/epoch-research")
	X86Prefix       string // image name prefix for x86 registry (e.g. "swe-bench.eval")
	MemLimit        string
	Client          *docker.Client
	Tmpfs           bool   // mount /testbed on tmpfs
	Runtime         string // container runtime ("crun" or "")
}

// Run loads data, evaluates predictions in parallel, and returns reports.
func Run(ctx context.Context, cfg Config) ([]grading.Report, error) {
	instances, err := dataset.LoadDataset(cfg.DatasetPath)
	if err != nil {
		return nil, fmt.Errorf("load dataset: %w", err)
	}

	predictions, err := dataset.LoadPredictions(cfg.PredictionsPath)
	if err != nil {
		return nil, fmt.Errorf("load predictions: %w", err)
	}

	tasks := dataset.Merge(instances, predictions)
	if len(tasks) == 0 {
		return nil, fmt.Errorf("no matching tasks found between dataset (%d instances) and predictions (%d predictions)",
			len(instances), len(predictions))
	}

	fmt.Printf("Matched %d tasks for evaluation (workers=%d, timeout=%v)\n", len(tasks), cfg.Workers, cfg.Timeout)

	pool := parallel.NewPool(cfg.Workers)

	results := parallel.Map(ctx, pool, tasks, func(ctx context.Context, task dataset.MergeTask) (grading.Report, error) {
		return evalTask(ctx, cfg, task)
	})

	var reports []grading.Report
	for _, r := range results {
		if r.Err != nil {
			reports = append(reports, grading.Report{
				Error: r.Err.Error(),
			})
		} else {
			reports = append(reports, r.Value)
		}
	}

	return reports, nil
}

// ResolveImage determines which Docker image and platform to use for an instance.
// Priority: local image > ARM64 registry (if set and instance is ARM64-compatible) > x86 registry.
func ResolveImage(ctx context.Context, cfg Config, inst dataset.Instance) (image string, platform string) {
	// Check for a locally-built image first
	imageSpec, specErr := spec.MakeImageSpec(inst, cfg.Arch)
	if specErr == nil {
		exists, err := docker.ImageExists(ctx, imageSpec.InstanceTag)
		if err == nil && exists {
			return imageSpec.InstanceTag, archToPlatform(cfg.Arch)
		}
	}

	// x86-only instances always use the x86 registry
	if spec.RequiresX86(inst.InstanceID) {
		return docker.ImageName(cfg.X86Registry, cfg.X86Prefix, inst.InstanceID, "x86_64"), "linux/amd64"
	}

	// Registry configured: use it for ARM64-compatible instances, selecting the
	// platform variant that matches the host arch (supports multi-arch manifests).
	if cfg.ARM64Registry != "" {
		tag := sanitizeName(inst.InstanceID)
		return fmt.Sprintf("%s:%s", cfg.ARM64Registry, tag), archToPlatform(cfg.Arch)
	}

	// Fallback: x86 registry (works everywhere, emulated on ARM64)
	return docker.ImageName(cfg.X86Registry, cfg.X86Prefix, inst.InstanceID, "x86_64"), "linux/amd64"
}

// buildCompoundScript combines patch application and eval into a single bash script.
// This saves 2-3 docker exec round-trips per evaluation.
func buildCompoundScript(inst dataset.Instance, pred dataset.Prediction) string {
	rs, err := spec.LookupSpec(inst.Repo, inst.Version)
	if err != nil {
		// Fallback to basic pytest if spec lookup fails
		rs = spec.RepoSpec{TestCmd: "pytest -rA"}
	}

	var b strings.Builder
	b.WriteString("#!/bin/bash\nset -xo pipefail\n\n")

	// Patch application with 3 fallback strategies
	b.WriteString("# Apply model patch\n")
	b.WriteString("cd /testbed\n")
	b.WriteString("PATCH_APPLIED=0\n")
	b.WriteString("if git apply -v /tmp/patch.diff 2>/dev/null; then\n")
	b.WriteString("  PATCH_APPLIED=1\n")
	b.WriteString("elif git apply -v --reject /tmp/patch.diff 2>/dev/null; then\n")
	b.WriteString("  PATCH_APPLIED=1\n")
	b.WriteString("elif patch -p1 -f < /tmp/patch.diff 2>/dev/null; then\n")
	b.WriteString("  PATCH_APPLIED=1\n")
	b.WriteString("fi\n\n")

	b.WriteString("if [ $PATCH_APPLIED -eq 0 ]; then\n")
	b.WriteString("  echo 'PATCH_APPLY_FAILED'\n")
	b.WriteString("  exit 1\n")
	b.WriteString("fi\n\n")

	b.WriteString("echo 'PATCH_APPLY_SUCCESS'\n\n")

	// Eval script body (without shebang)
	b.WriteString("# Eval\n")
	b.WriteString(eval.GenerateEvalScriptBody(inst, rs))

	return b.String()
}

func evalTask(ctx context.Context, cfg Config, task dataset.MergeTask) (grading.Report, error) {
	start := time.Now()
	inst := task.Instance
	pred := task.Prediction

	report := grading.Report{
		InstanceID: inst.InstanceID,
		Resolved:   grading.ResolvedNo,
	}

	image, platform := ResolveImage(ctx, cfg, inst)
	containerName := fmt.Sprintf("swe-bench-%s-%s", cfg.RunID, sanitizeName(inst.InstanceID))

	// Check if image exists locally before pulling
	exists, err := docker.ImageExists(ctx, image)
	if err != nil {
		report.Error = fmt.Sprintf("image check: %v", err)
		report.DurationMS = time.Since(start).Milliseconds()
		return report, nil
	}

	if !exists {
		if err := docker.Pull(ctx, image); err != nil {
			report.Error = fmt.Sprintf("pull: %v", err)
			report.DurationMS = time.Since(start).Milliseconds()
			return report, nil
		}
	}

	// Parse memory limit
	memBytes, err := docker.ParseMemLimit(cfg.MemLimit)
	if err != nil {
		memBytes = 4 * 1024 * 1024 * 1024 // 4GB fallback
	}

	// Look up spec for NanoCPUs
	rs, _ := spec.LookupSpec(inst.Repo, inst.Version)

	// Build container options with tmpfs and runtime support
	opts := docker.ContainerOpts{
		Name:     containerName,
		Image:    image,
		MemLimit: memBytes,
		NanoCPUs: rs.NanoCPUs,
		Network:  "none",
		Runtime:  cfg.Runtime,
		Platform: platform,
	}

	// tmpfs on /testbed blanks the pre-installed repo in both Epoch and custom
	// images (content is baked into image layers). Only safe for images where
	// /testbed is populated at runtime, which we don't do yet.
	// TODO: support tmpfs by copying /testbed into tmpfs at container start.

	// Create and start container
	containerID, err := docker.CreateWithOpts(ctx, opts)
	if err != nil {
		report.Error = fmt.Sprintf("create: %v", err)
		report.DurationMS = time.Since(start).Milliseconds()
		return report, nil
	}
	defer docker.Remove(ctx, containerID)

	if err := docker.Start(ctx, containerID); err != nil {
		report.Error = fmt.Sprintf("start: %v", err)
		report.DurationMS = time.Since(start).Milliseconds()
		return report, nil
	}

	// Copy prediction patch into container
	if err := docker.CopyTo(ctx, containerID, pred.ModelPatch, "/tmp/patch.diff"); err != nil {
		report.Error = fmt.Sprintf("copy patch: %v", err)
		report.DurationMS = time.Since(start).Milliseconds()
		return report, nil
	}

	// Generate and copy compound eval script
	compoundScript := buildCompoundScript(inst, pred)
	if err := docker.CopyTo(ctx, containerID, compoundScript, "/tmp/eval_compound.sh"); err != nil {
		report.Error = fmt.Sprintf("copy compound script: %v", err)
		report.DurationMS = time.Since(start).Milliseconds()
		return report, nil
	}

	// Execute compound script (single exec replaces patch + eval)
	stdout, stderr, _, err := docker.Exec(ctx, containerID, "/bin/bash /tmp/eval_compound.sh", cfg.Timeout)
	if err != nil {
		report.Error = fmt.Sprintf("eval exec: %v", err)
		report.DurationMS = time.Since(start).Milliseconds()
		return report, nil
	}

	// Check if patch was applied successfully
	fullLog := stdout + "\n" + stderr
	patchApplied := strings.Contains(fullLog, "PATCH_APPLY_SUCCESS")

	report.PatchApplied = patchApplied
	if !patchApplied {
		// Include truncated log in error for debugging
		logSnip := fullLog
		if len(logSnip) > 2000 {
			logSnip = logSnip[:2000]
		}
		report.Error = fmt.Sprintf("patch apply failed; log: %s", logSnip)
		report.DurationMS = time.Since(start).Milliseconds()
		return report, nil
	}

	// Parse test output
	statusMap := parser.ParseLog(inst.Repo, fullLog)

	if len(statusMap) == 0 {
		// No test results parsed — likely the eval script exited before tests ran.
		logSnip := fullLog
		if len(logSnip) > 2000 {
			logSnip = logSnip[len(logSnip)-2000:]
		}
		fmt.Printf("WARNING [%s]: 0 tests parsed from output (%d bytes); tail: %s\n",
			inst.InstanceID, len(fullLog), logSnip)
	}

	// Grade results
	result := grading.Grade(statusMap, inst.FailToPass, inst.PassToPass)
	result.InstanceID = inst.InstanceID
	result.PatchApplied = patchApplied
	result.DurationMS = time.Since(start).Milliseconds()

	return result, nil
}

// archToPlatform converts a server architecture string to a Docker platform string.
func archToPlatform(arch string) string {
	if arch == "aarch64" || arch == "arm64" {
		return "linux/arm64"
	}
	return "linux/amd64"
}

// sanitizeName replaces characters not valid in Docker container names.
func sanitizeName(s string) string {
	r := strings.NewReplacer("/", "-", "__", "-", ".", "-")
	return r.Replace(s)
}

// Summary holds aggregate statistics for a run.
type Summary struct {
	Total       int     `json:"total"`
	Resolved    int     `json:"resolved"`
	Partial     int     `json:"partial"`
	Unresolved  int     `json:"unresolved"`
	Errors      int     `json:"errors"`
	ResolvedPct float64 `json:"resolved_pct"`
	TotalTimeMS int64   `json:"total_time_ms"`
}

// Summarize computes aggregate statistics from a slice of reports.
func Summarize(reports []grading.Report) Summary {
	var s Summary
	s.Total = len(reports)
	for _, r := range reports {
		s.TotalTimeMS += r.DurationMS
		switch {
		case r.Error != "":
			s.Errors++
		case r.Resolved == grading.ResolvedFull:
			s.Resolved++
		case r.Resolved == grading.ResolvedPartial:
			s.Partial++
		default:
			s.Unresolved++
		}
	}
	if s.Total > 0 {
		s.ResolvedPct = float64(s.Resolved) / float64(s.Total) * 100
	}
	return s
}
