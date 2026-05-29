package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/greynewell/mist-go/cli"
	"github.com/greynewell/mist-go/config"
	"github.com/greynewell/mist-go/lifecycle"
	"github.com/greynewell/mist-go/output"

	"github.com/greynewell/swe-bench-fast/internal/builder"
	"github.com/greynewell/swe-bench-fast/internal/dataset"
	"github.com/greynewell/swe-bench-fast/internal/docker"
	"github.com/greynewell/swe-bench-fast/internal/grading"
	"github.com/greynewell/swe-bench-fast/internal/runner"
	"github.com/greynewell/swe-bench-fast/internal/spec"
)

var version = "dev"

// appConfig mirrors the swe-bench-fast.toml structure.
type appConfig struct {
	Name          string `toml:"name"`
	Workers       int    `toml:"workers"`
	Timeout       int    `toml:"timeout"`
	Arch          string `toml:"arch"`
	CheckpointDir string `toml:"checkpoint_dir"`
	ARM64Registry string `toml:"arm64_registry"`
	X86Registry   string `toml:"x86_registry"`
	X86Prefix     string `toml:"x86_prefix"`
	MemLimit      string `toml:"mem_limit"`
	Tmpfs         bool   `toml:"tmpfs"`
	Runtime       string `toml:"runtime"`
	BuildWorkers  int    `toml:"build_workers"`
}

func main() {
	app := cli.NewApp("swe-bench-fast", version)

	app.AddCommand(runCmd())
	app.AddCommand(buildCmd())
	app.AddCommand(validateCmd())
	app.AddCommand(versionCmd())

	if err := app.Execute(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func defaultArch() string {
	switch runtime.GOARCH {
	case "arm64":
		return "aarch64"
	default:
		return "x86_64"
	}
}

func loadConfig() appConfig {
	cfg := appConfig{
		Workers:       8,
		Timeout:       300,
		Arch:          defaultArch(),
		CheckpointDir: ".checkpoints",
		ARM64Registry: "",
		X86Registry:   "ghcr.io/epoch-research",
		X86Prefix:     "swe-bench.eval",
		MemLimit:      "4g",
		Tmpfs:         runtime.GOOS == "linux",
		BuildWorkers:  4,
	}
	_ = config.Load("swe-bench-fast.toml", "SWE_BENCH_FAST", &cfg)
	return cfg
}

func runCmd() *cli.Command {
	cmd := &cli.Command{
		Name:  "run",
		Usage: "Evaluate predictions against SWE-bench instances",
	}

	cmd.AddStringFlag("dataset", "", "Path to SWE-bench dataset (JSON or JSONL)")
	cmd.AddStringFlag("predictions", "", "Path to predictions JSONL file")
	cmd.AddIntFlag("workers", 0, "Number of parallel workers (0 = use config)")
	cmd.AddIntFlag("timeout", 0, "Per-task timeout in seconds (0 = use config)")
	cmd.AddStringFlag("format", "table", "Output format: table or json")
	cmd.AddStringFlag("run-id", "", "Run identifier for checkpointing")
	cmd.AddStringFlag("output", "", "Path to write JSON report (optional)")
	cmd.AddBoolFlag("tmpfs", false, "Mount /testbed on tmpfs (default: true on Linux)")
	cmd.AddStringFlag("runtime", "", "Container runtime (e.g. crun)")

	cmd.Run = func(cmd *cli.Command, args []string) error {
		cfg := loadConfig()

		datasetPath := cmd.GetString("dataset")
		if datasetPath == "" {
			return fmt.Errorf("--dataset is required")
		}

		predsPath := cmd.GetString("predictions")
		if predsPath == "" {
			return fmt.Errorf("--predictions is required")
		}

		workers := cmd.GetInt("workers")
		if workers == 0 {
			workers = cfg.Workers
		}

		timeout := cmd.GetInt("timeout")
		if timeout == 0 {
			timeout = cfg.Timeout
		}

		runID := cmd.GetString("run-id")
		if runID == "" {
			runID = fmt.Sprintf("run-%d", time.Now().Unix())
		}

		useTmpfs := cmd.GetBool("tmpfs") || cfg.Tmpfs
		runtimeStr := cmd.GetString("runtime")
		if runtimeStr == "" {
			runtimeStr = cfg.Runtime
		}

		client, err := docker.NewClient()
		if err != nil {
			return fmt.Errorf("docker client: %w", err)
		}
		defer client.Close()
		docker.SetDefaultClient(client)

		runCfg := runner.Config{
			DatasetPath:     datasetPath,
			PredictionsPath: predsPath,
			Workers:         workers,
			Timeout:         time.Duration(timeout) * time.Second,
			RunID:           runID,
			Arch:            cfg.Arch,
			ARM64Registry:   cfg.ARM64Registry,
			X86Registry:     cfg.X86Registry,
			X86Prefix:       cfg.X86Prefix,
			MemLimit:        cfg.MemLimit,
			Client:          client,
			Tmpfs:           useTmpfs,
			Runtime:         runtimeStr,
		}

		return lifecycle.Run(func(ctx context.Context) error {
			reports, err := runner.Run(ctx, runCfg)
			if err != nil {
				return err
			}

			summary := runner.Summarize(reports)

			// Write output
			format := cmd.GetString("format")
			switch format {
			case "json":
				printJSON(reports, summary)
			default:
				printTable(reports, summary)
			}

			// Optionally write JSON report to file
			if outPath := cmd.GetString("output"); outPath != "" {
				if err := writeReport(outPath, reports, summary); err != nil {
					return fmt.Errorf("write report: %w", err)
				}
				fmt.Printf("\nReport written to %s\n", outPath)
			}

			return nil
		})
	}

	return cmd
}

func buildCmd() *cli.Command {
	cmd := &cli.Command{
		Name:  "build",
		Usage: "Build custom Docker images for SWE-bench instances",
	}

	cmd.AddStringFlag("dataset", "", "Path to SWE-bench dataset (JSON or JSONL)")
	cmd.AddIntFlag("workers", 0, "Number of parallel build workers (0 = use config)")
	cmd.AddBoolFlag("force", false, "Force rebuild even if images exist")
	cmd.AddStringFlag("arch", "", "Architecture (x86_64 or aarch64)")
	cmd.AddStringFlag("push", "", "Push instance images to this registry and delete locally (e.g. ghcr.io/user)")

	cmd.Run = func(cmd *cli.Command, args []string) error {
		cfg := loadConfig()

		datasetPath := cmd.GetString("dataset")
		if datasetPath == "" {
			return fmt.Errorf("--dataset is required")
		}

		workers := cmd.GetInt("workers")
		if workers == 0 {
			workers = cfg.BuildWorkers
		}

		forceRebuild := cmd.GetBool("force")
		pushRegistry := cmd.GetString("push")

		// Initialize Docker SDK client
		client, err := docker.NewClient()
		if err != nil {
			return fmt.Errorf("docker client: %w", err)
		}
		defer client.Close()

		arch := cmd.GetString("arch")
		if arch == "" {
			// Auto-detect from Docker daemon
			arch = client.ServerArch(context.Background())
			fmt.Printf("Auto-detected Docker architecture: %s\n", arch)
		}

		return lifecycle.Run(func(ctx context.Context) error {
			// Load dataset
			instances, err := dataset.LoadDataset(datasetPath)
			if err != nil {
				return fmt.Errorf("load dataset: %w", err)
			}

			fmt.Printf("Loaded %d instances\n", len(instances))

			// Create image specs
			var specs []spec.ImageSpec
			var skipped int
			for _, inst := range instances {
				imageSpec, err := spec.MakeImageSpec(inst, arch)
				if err != nil {
					skipped++
					continue
				}
				specs = append(specs, imageSpec)
			}

			if skipped > 0 {
				fmt.Printf("Skipped %d instances (unknown repo/version)\n", skipped)
			}
			fmt.Printf("Building images for %d instances\n", len(specs))

			// Build all images
			buildCfg := builder.BuildConfig{
				Client:       client,
				Workers:      workers,
				ForceRebuild: forceRebuild,
				Arch:         arch,
			}

			// If --push is set, push each instance image to the registry and delete locally
			if pushRegistry != "" {
				fmt.Printf("Push mode: images will be pushed to %s and deleted locally\n", pushRegistry)
				var pushMu sync.Mutex
				var pushed, pushErrors int
				buildCfg.PostBuild = func(ctx context.Context, s spec.ImageSpec) error {
					// Tag: repo:instance-name (e.g. greynewell/swe-bench-fast:astropy-astropy-12907)
					localTag := s.InstanceTag
					suffix := strings.TrimPrefix(localTag, "swe-bench-instance:")
					remoteTag := fmt.Sprintf("%s:%s", pushRegistry, suffix)

					if err := client.ImageTag(ctx, localTag, remoteTag); err != nil {
						pushMu.Lock()
						pushErrors++
						pushMu.Unlock()
						return fmt.Errorf("tag %s: %w", remoteTag, err)
					}

					fmt.Printf("  [push]    %s\n", remoteTag)
					if err := client.ImagePush(ctx, remoteTag); err != nil {
						pushMu.Lock()
						pushErrors++
						pushMu.Unlock()
						return fmt.Errorf("push %s: %w", remoteTag, err)
					}

					// Delete local instance image (keep env cached)
					_ = client.ImageRemove(ctx, remoteTag)
					_ = client.ImageRemove(ctx, localTag)
					fmt.Printf("  [delete]  %s\n", localTag)

					pushMu.Lock()
					pushed++
					pushMu.Unlock()
					return nil
				}
			}

			buildStart := time.Now()
			results, err := builder.BuildAll(ctx, buildCfg, specs)
			buildElapsed := time.Since(buildStart)
			if err != nil {
				return fmt.Errorf("build: %w", err)
			}

			// Report results
			var built, cached, errors int
			for _, r := range results {
				switch {
				case r.Error != nil:
					errors++
					fmt.Printf("  ERROR %s: %v\n", r.Tag, r.Error)
				case r.Cached:
					cached++
					fmt.Printf("  [cached]  %s (%s)\n", r.Tag, r.Duration.Round(time.Millisecond))
				default:
					built++
					fmt.Printf("  [built]   %s (%s)\n", r.Tag, r.Duration.Round(time.Second))
				}
			}

			fmt.Printf("\n--- Build Summary ---\n")
			fmt.Printf("Built: %d | Cached: %d | Errors: %d | Total: %d\n",
				built, cached, errors, len(results))
			fmt.Printf("Wall time: %s\n", buildElapsed.Round(time.Second))

			return nil
		})
	}

	return cmd
}

func validateCmd() *cli.Command {
	cmd := &cli.Command{
		Name:  "validate",
		Usage: "Smoke-test Docker images before full evaluation",
	}

	cmd.AddStringFlag("dataset", "", "Path to SWE-bench dataset (JSON or JSONL)")
	cmd.AddBoolFlag("smoke", false, "Run smoke test (conda activate, pytest --version) in each container")

	cmd.Run = func(cmd *cli.Command, args []string) error {
		cfg := loadConfig()

		datasetPath := cmd.GetString("dataset")
		if datasetPath == "" {
			return fmt.Errorf("--dataset is required")
		}

		client, err := docker.NewClient()
		if err != nil {
			return fmt.Errorf("docker client: %w", err)
		}
		defer client.Close()
		docker.SetDefaultClient(client)

		runSmoke := cmd.GetBool("smoke")

		return lifecycle.Run(func(ctx context.Context) error {
			instances, err := dataset.LoadDataset(datasetPath)
			if err != nil {
				return fmt.Errorf("load dataset: %w", err)
			}

			fmt.Printf("Validating %d instances...\n", len(instances))

			rCfg := runner.Config{
				Arch:          cfg.Arch,
				ARM64Registry: cfg.ARM64Registry,
				X86Registry:   cfg.X86Registry,
				X86Prefix:     cfg.X86Prefix,
			}

			var passed, missing, failed int
			for _, inst := range instances {
				image, platform := runner.ResolveImage(ctx, rCfg, inst)

				exists, err := docker.ImageExists(ctx, image)
				if err != nil {
					fmt.Printf("  ERROR %s: %v\n", inst.InstanceID, err)
					failed++
					continue
				}
				if !exists {
					fmt.Printf("  MISSING %s: %s\n", inst.InstanceID, image)
					missing++
					continue
				}

				if !runSmoke {
					fmt.Printf("  OK %s: %s (%s)\n", inst.InstanceID, image, platform)
					passed++
					continue
				}

				if err := smokeTest(ctx, image, inst.InstanceID, cfg.MemLimit); err != nil {
					fmt.Printf("  FAIL %s: %v\n", inst.InstanceID, err)
					failed++
					continue
				}

				fmt.Printf("  OK %s: smoke test passed (%s)\n", inst.InstanceID, platform)
				passed++
			}

			fmt.Printf("\n--- Validate Summary ---\n")
			fmt.Printf("Passed: %d | Missing: %d | Failed: %d | Total: %d\n",
				passed, missing, failed, len(instances))

			if missing+failed > 0 {
				return fmt.Errorf("%d images missing, %d smoke tests failed", missing, failed)
			}
			return nil
		})
	}

	return cmd
}

func smokeTest(ctx context.Context, image, instanceID, memLimit string) error {
	memBytes, err := docker.ParseMemLimit(memLimit)
	if err != nil {
		memBytes = 4 * 1024 * 1024 * 1024
	}

	containerName := fmt.Sprintf("swe-validate-%s", sanitizeName(instanceID))

	containerID, err := docker.CreateWithOpts(ctx, docker.ContainerOpts{
		Name:     containerName,
		Image:    image,
		MemLimit: memBytes,
		Network:  "none",
	})
	if err != nil {
		return fmt.Errorf("create container: %w", err)
	}
	defer docker.Remove(ctx, containerID)

	if err := docker.Start(ctx, containerID); err != nil {
		return fmt.Errorf("start container: %w", err)
	}

	// Check conda activate and pytest --version
	script := `set +u
source /opt/miniconda3/bin/activate
conda activate testbed
set -u
cd /testbed
pytest --version
python -c "import sys; print('Python', sys.version)"
`
	stdout, stderr, exitCode, err := docker.Exec(ctx, containerID, script, 30*time.Second)
	if err != nil {
		return fmt.Errorf("exec: %w", err)
	}
	if exitCode != 0 {
		// Show last 500 chars of output for debugging
		combined := stdout + "\n" + stderr
		if len(combined) > 500 {
			combined = combined[len(combined)-500:]
		}
		return fmt.Errorf("exit code %d: %s", exitCode, combined)
	}
	return nil
}

func sanitizeName(s string) string {
	r := strings.NewReplacer("/", "-", "__", "-", ".", "-")
	return r.Replace(s)
}

func versionCmd() *cli.Command {
	cmd := &cli.Command{
		Name:  "version",
		Usage: "Print version information",
	}
	cmd.Run = func(cmd *cli.Command, args []string) error {
		fmt.Printf("swe-bench-fast %s\n", version)
		return nil
	}
	return cmd
}

func printTable(reports []grading.Report, summary runner.Summary) {
	w := output.New("table")

	headers := []string{"Instance", "Resolved", "F2P", "P2P", "Patch", "Time(s)", "Error"}
	var rows [][]string

	for _, r := range reports {
		patchStr := "no"
		if r.PatchApplied {
			patchStr = "yes"
		}
		errStr := r.Error
		if len(errStr) > 50 {
			errStr = errStr[:50] + "..."
		}
		rows = append(rows, []string{
			r.InstanceID,
			string(r.Resolved),
			fmt.Sprintf("%d/%d", r.F2PPassed, r.F2PTotal),
			fmt.Sprintf("%d/%d", r.P2PPassed, r.P2PTotal),
			patchStr,
			fmt.Sprintf("%.1f", float64(r.DurationMS)/1000),
			errStr,
		})
	}

	w.Table(headers, rows)

	fmt.Printf("\n--- Summary ---\n")
	fmt.Printf("Total: %d | Resolved: %d (%.1f%%) | Partial: %d | Unresolved: %d | Errors: %d\n",
		summary.Total, summary.Resolved, summary.ResolvedPct, summary.Partial, summary.Unresolved, summary.Errors)
}

func printJSON(reports []grading.Report, summary runner.Summary) {
	w := output.New("json")
	result := map[string]any{
		"reports": reports,
		"summary": summary,
	}
	w.JSON(result)
}

type fullReport struct {
	Reports []grading.Report `json:"reports"`
	Summary runner.Summary   `json:"summary"`
}

func writeReport(path string, reports []grading.Report, summary runner.Summary) error {
	data, err := json.MarshalIndent(fullReport{Reports: reports, Summary: summary}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}
