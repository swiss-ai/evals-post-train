package spec

import (
	"crypto/sha256"
	"fmt"
	"strings"

	"github.com/greynewell/swe-bench-fast/internal/dataset"
)

// RepoSpec describes the environment and test configuration for a repo version.
type RepoSpec struct {
	Python        string   // e.g. "3.9", "3.11"
	Packages      string   // "requirements.txt", "environment.yml", or conda packages
	PipPackages   []string // pinned pip packages to install
	PreInstall    []string // shell commands to run before install
	Install       string   // install command, e.g. "python -m pip install -e ."
	TestCmd       string   // test runner command, e.g. "pytest -rA"
	EvalEnvVars   []string // commands run before test during eval (exports, locale setup, etc.)
	NanoCPUs      int64    // CPU limit for container (0=unlimited, 2e9=2 cores)
	NoUseEnv      bool     // skip "conda activate testbed" in eval script
	ExecAsNonRoot bool     // run tests as non-root user
}

// ImageSpec describes the 3-layer image build for a single instance.
type ImageSpec struct {
	InstanceID  string
	Repo        string
	Version     string
	BaseCommit  string
	Arch        string
	RepoSpec    RepoSpec
	BaseTag     string // deterministic tag for base layer
	EnvTag      string // deterministic tag for env layer
	InstanceTag string // deterministic tag for instance layer
}

// repoSpecs maps repo → version → RepoSpec for all repos in SWE-bench.
// Populated by init() in specs_generated.go.
var repoSpecs = map[string]map[string]RepoSpec{}

// registerSpecs registers specs for a given repo.
func registerSpecs(repo string, specs map[string]RepoSpec) {
	repoSpecs[repo] = specs
}

// LookupSpec returns the RepoSpec for a given repo and version.
func LookupSpec(repo, version string) (RepoSpec, error) {
	repoMap, ok := repoSpecs[repo]
	if !ok {
		return RepoSpec{}, fmt.Errorf("unknown repo: %s", repo)
	}
	spec, ok := repoMap[version]
	if !ok {
		return RepoSpec{}, fmt.Errorf("unknown version %s for repo %s", version, repo)
	}
	return spec, nil
}

// MakeImageSpec creates the full ImageSpec for an instance, computing deterministic tags.
func MakeImageSpec(inst dataset.Instance, arch string) (ImageSpec, error) {
	rs, err := LookupSpec(inst.Repo, inst.Version)
	if err != nil {
		return ImageSpec{}, err
	}

	baseTag := computeBaseTag(arch)
	envTag := computeEnvTag(baseTag, inst.Repo, inst.Version, rs)
	instanceTag := computeInstanceTag(inst.InstanceID)

	return ImageSpec{
		InstanceID:  inst.InstanceID,
		Repo:        inst.Repo,
		Version:     inst.Version,
		BaseCommit:  inst.BaseCommit,
		Arch:        arch,
		RepoSpec:    rs,
		BaseTag:     baseTag,
		EnvTag:      envTag,
		InstanceTag: instanceTag,
	}, nil
}

func computeBaseTag(arch string) string {
	h := sha256.Sum256([]byte("swe-bench-base:" + arch))
	return fmt.Sprintf("swe-bench-base:%s", hex8(h))
}

func computeEnvTag(baseTag, repo, version string, rs RepoSpec) string {
	input := fmt.Sprintf("%s:%s:%s:%s:%s:%s",
		baseTag, repo, version, rs.Python, rs.Packages, strings.Join(rs.PipPackages, ","))
	h := sha256.Sum256([]byte(input))
	return fmt.Sprintf("swe-bench-env:%s", hex8(h))
}

func computeInstanceTag(instanceID string) string {
	// Instance tags are directly based on the instance ID for readability
	safe := strings.NewReplacer("/", "-", "__", "-").Replace(instanceID)
	return fmt.Sprintf("swe-bench-instance:%s", strings.ToLower(safe))
}

func hex8(h [32]byte) string {
	return fmt.Sprintf("%x", h[:8])
}
