package spec

import (
	"strings"
	"testing"

	"github.com/greynewell/swe-bench-fast/internal/dataset"
)

func TestLookupSpec(t *testing.T) {
	tests := []struct {
		repo    string
		version string
		wantPy  string
		wantErr bool
	}{
		// Django: Python versions per upstream
		{"django/django", "1.7", "3.5", false},
		{"django/django", "3.0", "3.6", false},
		{"django/django", "4.0", "3.8", false},
		{"django/django", "4.2", "3.9", false},
		{"django/django", "5.0", "3.11", false},
		// Pytest
		{"pytest-dev/pytest", "5.0", "3.9", false},
		{"pytest-dev/pytest", "7.0", "3.9", false},
		{"pytest-dev/pytest", "8.4", "3.9", false},
		// Scikit-learn: 0.20 uses Python 3.6
		{"scikit-learn/scikit-learn", "0.20", "3.6", false},
		{"scikit-learn/scikit-learn", "1.3", "3.9", false},
		// Matplotlib: 3.0 → 3.7, 3.5 → 3.11
		{"matplotlib/matplotlib", "3.0", "3.7", false},
		{"matplotlib/matplotlib", "3.5", "3.11", false},
		// Sympy
		{"sympy/sympy", "1.12", "3.9", false},
		// Astropy: v5.3 → 3.10
		{"astropy/astropy", "5.0", "3.9", false},
		{"astropy/astropy", "v5.3", "3.10", false},
		// Flask: 2.1 → 3.10, 2.2 → 3.11
		{"pallets/flask", "2.0", "3.9", false},
		{"pallets/flask", "2.1", "3.10", false},
		{"pallets/flask", "2.2", "3.11", false},
		// Sphinx
		{"sphinx-doc/sphinx", "4.0", "3.9", false},
		{"sphinx-doc/sphinx", "8.0", "3.10", false},
		// Requests
		{"psf/requests", "2.25", "3.9", false},
		// Xarray: Python 3.10
		{"pydata/xarray", "2022.03", "3.10", false},
		// Seaborn
		{"mwaskom/seaborn", "0.12", "3.9", false},
		// Marshmallow
		{"marshmallow-code/marshmallow", "3.0", "3.9", false},
		// Pylint
		{"pylint-dev/pylint", "2.13", "3.9", false},
		// Pydicom: per-version Python
		{"pydicom/pydicom", "1.0", "3.6", false},
		{"pydicom/pydicom", "2.0", "3.8", false},
		{"pydicom/pydicom", "2.3", "3.10", false},
		{"pydicom/pydicom", "3.0", "3.11", false},
		// New repos
		{"pylint-dev/astroid", "2.5", "3.9", false},
		{"sqlfluff/sqlfluff", "1.0", "3.9", false},
		{"dbt-labs/dbt-core", "1.0", "3.9", false},
		{"pyvista/pyvista", "0.30", "3.9", false},
		{"pvlib/pvlib-python", "0.5", "3.9", false},
		{"swe-bench/humaneval", "1.0", "3.9", false},
		// Error cases
		{"unknown/repo", "1.0", "", true},
		{"django/django", "99.99", "", true},
	}

	for _, tt := range tests {
		rs, err := LookupSpec(tt.repo, tt.version)
		if (err != nil) != tt.wantErr {
			t.Errorf("LookupSpec(%s, %s) error = %v, wantErr %v", tt.repo, tt.version, err, tt.wantErr)
			continue
		}
		if err == nil && rs.Python != tt.wantPy {
			t.Errorf("LookupSpec(%s, %s).Python = %q, want %q", tt.repo, tt.version, rs.Python, tt.wantPy)
		}
	}
}

func TestLookupSpecInstallCommand(t *testing.T) {
	// Verify all specs have required fields (except humaneval which is special)
	for repo, versions := range repoSpecs {
		for version, rs := range versions {
			if rs.Python == "" {
				t.Errorf("repoSpecs[%s][%s].Python is empty", repo, version)
			}
			if rs.TestCmd == "" {
				t.Errorf("repoSpecs[%s][%s].TestCmd is empty", repo, version)
			}
			// humaneval doesn't need Install
			if repo != "swe-bench/humaneval" && rs.Install == "" {
				t.Errorf("repoSpecs[%s][%s].Install is empty", repo, version)
			}
		}
	}
}

func TestLookupSpecTestCmds(t *testing.T) {
	// Verify key test commands match upstream
	tests := []struct {
		repo    string
		version string
		wantCmd string
	}{
		{"django/django", "3.0", testDjango},
		{"django/django", "1.9", testDjangoNoParallel},
		{"pytest-dev/pytest", "8.0", testPytest},
		{"sympy/sympy", "1.0", testSympy},
		{"sphinx-doc/sphinx", "4.0", testSphinx},
		{"mwaskom/seaborn", "0.12", testSeaborn},
		{"astropy/astropy", "0.1", testAstropyPytest},
		{"swe-bench/humaneval", "1.0", "python"},
	}
	for _, tt := range tests {
		rs, err := LookupSpec(tt.repo, tt.version)
		if err != nil {
			t.Errorf("LookupSpec(%s, %s) error = %v", tt.repo, tt.version, err)
			continue
		}
		if rs.TestCmd != tt.wantCmd {
			t.Errorf("LookupSpec(%s, %s).TestCmd = %q, want %q", tt.repo, tt.version, rs.TestCmd, tt.wantCmd)
		}
	}
}

func TestLookupSpecPylintNanoCPUs(t *testing.T) {
	rs, err := LookupSpec("pylint-dev/pylint", "3.0")
	if err != nil {
		t.Fatalf("LookupSpec error: %v", err)
	}
	if rs.NanoCPUs != int64(2e9) {
		t.Errorf("pylint 3.0 NanoCPUs = %d, want %d", rs.NanoCPUs, int64(2e9))
	}
}

func TestLookupSpecXarrayNoUseEnv(t *testing.T) {
	rs, err := LookupSpec("pydata/xarray", "2022.03")
	if err != nil {
		t.Fatalf("LookupSpec error: %v", err)
	}
	if !rs.NoUseEnv {
		t.Error("xarray should have NoUseEnv=true")
	}
}

func TestMakeImageSpec(t *testing.T) {
	inst := dataset.Instance{
		InstanceID: "django__django-11099",
		Repo:       "django/django",
		BaseCommit: "abc123",
		Version:    "3.0",
	}

	imageSpec, err := MakeImageSpec(inst, "x86_64")
	if err != nil {
		t.Fatalf("MakeImageSpec() error: %v", err)
	}

	if imageSpec.BaseTag == "" {
		t.Error("BaseTag is empty")
	}
	if imageSpec.EnvTag == "" {
		t.Error("EnvTag is empty")
	}
	if imageSpec.InstanceTag == "" {
		t.Error("InstanceTag is empty")
	}
	if !strings.HasPrefix(imageSpec.BaseTag, "swe-bench-base:") {
		t.Errorf("BaseTag = %q, want prefix 'swe-bench-base:'", imageSpec.BaseTag)
	}
	if !strings.HasPrefix(imageSpec.EnvTag, "swe-bench-env:") {
		t.Errorf("EnvTag = %q, want prefix 'swe-bench-env:'", imageSpec.EnvTag)
	}
	if !strings.HasPrefix(imageSpec.InstanceTag, "swe-bench-instance:") {
		t.Errorf("InstanceTag = %q, want prefix 'swe-bench-instance:'", imageSpec.InstanceTag)
	}
}

func TestMakeImageSpecDeterministic(t *testing.T) {
	inst := dataset.Instance{
		InstanceID: "django__django-11099",
		Repo:       "django/django",
		BaseCommit: "abc123",
		Version:    "3.0",
	}

	spec1, _ := MakeImageSpec(inst, "x86_64")
	spec2, _ := MakeImageSpec(inst, "x86_64")

	if spec1.BaseTag != spec2.BaseTag {
		t.Error("BaseTag not deterministic")
	}
	if spec1.EnvTag != spec2.EnvTag {
		t.Error("EnvTag not deterministic")
	}
	if spec1.InstanceTag != spec2.InstanceTag {
		t.Error("InstanceTag not deterministic")
	}
}

func TestMakeImageSpecUnknownRepo(t *testing.T) {
	inst := dataset.Instance{
		InstanceID: "unknown__repo-1",
		Repo:       "unknown/repo",
		Version:    "1.0",
	}

	_, err := MakeImageSpec(inst, "x86_64")
	if err == nil {
		t.Error("expected error for unknown repo")
	}
}

func TestGenerateSetupEnvScript(t *testing.T) {
	inst := dataset.Instance{
		InstanceID: "django__django-11099",
		Repo:       "django/django",
		BaseCommit: "abc123",
		Version:    "3.0",
	}

	imageSpec, err := MakeImageSpec(inst, "x86_64")
	if err != nil {
		t.Fatalf("MakeImageSpec error: %v", err)
	}

	script := GenerateSetupEnvScript(imageSpec)
	if !strings.Contains(script, "conda create") {
		t.Error("missing conda create")
	}
	if !strings.Contains(script, "python=3.6") {
		t.Errorf("missing python version, got:\n%s", script)
	}
}

func TestGenerateSetupEnvScriptCondaPackages(t *testing.T) {
	inst := dataset.Instance{
		InstanceID: "scikit-learn__scikit-learn-12345",
		Repo:       "scikit-learn/scikit-learn",
		BaseCommit: "abc123",
		Version:    "0.20",
	}

	imageSpec, err := MakeImageSpec(inst, "x86_64")
	if err != nil {
		t.Fatalf("MakeImageSpec error: %v", err)
	}

	script := GenerateSetupEnvScript(imageSpec)
	// Upstream puts inline packages in the conda create command
	if !strings.Contains(script, "conda create -n testbed python=3.6 numpy scipy cython pytest pandas matplotlib -y") {
		t.Errorf("missing conda create with sklearn packages, got:\n%s", script)
	}
}

func TestGenerateSetupRepoScript(t *testing.T) {
	inst := dataset.Instance{
		InstanceID: "django__django-11099",
		Repo:       "django/django",
		BaseCommit: "abc123",
		Version:    "3.0",
	}

	imageSpec, err := MakeImageSpec(inst, "x86_64")
	if err != nil {
		t.Fatalf("MakeImageSpec error: %v", err)
	}

	script := GenerateSetupRepoScript(imageSpec)
	if !strings.Contains(script, "git clone") {
		t.Error("missing git clone")
	}
	if !strings.Contains(script, "git reset --hard abc123") {
		t.Error("missing git reset --hard")
	}
	if !strings.Contains(script, "pip install -e .") {
		t.Error("missing install command")
	}
	// Django 3.0 uses requirements.txt → should install from repo requirements
	if !strings.Contains(script, "pip install -r tests/requirements/py3.txt") {
		t.Errorf("missing requirements.txt install for Django, got:\n%s", script)
	}
}

func TestDockerfileGeneration(t *testing.T) {
	base := GenerateBaseDockerfile("x86_64")
	if !strings.Contains(base, "ubuntu:22.04") {
		t.Error("base dockerfile missing ubuntu base")
	}
	if !strings.Contains(base, "miniconda") {
		t.Error("base dockerfile missing miniconda")
	}
	if !strings.Contains(base, "x86_64") {
		t.Error("base dockerfile missing x86_64 arch")
	}

	// arm64 variant
	baseArm := GenerateBaseDockerfile("aarch64")
	if !strings.Contains(baseArm, "aarch64") {
		t.Error("arm64 base dockerfile missing aarch64 arch")
	}

	env := GenerateEnvDockerfile("swe-bench-base:abc123")
	if !strings.Contains(env, "FROM swe-bench-base:abc123") {
		t.Error("env dockerfile missing FROM base tag")
	}
	if !strings.Contains(env, "setup_env.sh") {
		t.Error("env dockerfile missing setup_env.sh")
	}

	instance := GenerateInstanceDockerfile("swe-bench-env:def456")
	if !strings.Contains(instance, "FROM swe-bench-env:def456") {
		t.Error("instance dockerfile missing FROM env tag")
	}
	if !strings.Contains(instance, "setup_repo.sh") {
		t.Error("instance dockerfile missing setup_repo.sh")
	}
}

// TestAllSWEBenchRepos verifies that all repos in upstream SWE-bench
// have at least one version configured.
func TestAllSWEBenchRepos(t *testing.T) {
	expectedRepos := []string{
		"astropy/astropy",
		"dbt-labs/dbt-core",
		"django/django",
		"flask/flask", // legacy alias for pallets/flask
		"marshmallow-code/marshmallow",
		"matplotlib/matplotlib",
		"mwaskom/seaborn",
		"pallets/flask",
		"psf/requests",
		"pvlib/pvlib-python",
		"pydata/xarray",
		"pydicom/pydicom",
		"pylint-dev/astroid",
		"pylint-dev/pylint",
		"pytest-dev/pytest",
		"pyvista/pyvista",
		"scikit-learn/scikit-learn",
		"sphinx-doc/sphinx",
		"sqlfluff/sqlfluff",
		"swe-bench/humaneval",
		"sympy/sympy",
	}

	for _, repo := range expectedRepos {
		versions, ok := repoSpecs[repo]
		if !ok {
			t.Errorf("missing repo: %s", repo)
			continue
		}
		if len(versions) == 0 {
			t.Errorf("repo %s has no versions", repo)
		}
	}

	// Verify total repo count matches upstream (20 repos + 1 legacy alias)
	if len(repoSpecs) != 21 {
		t.Errorf("expected 21 repos, got %d", len(repoSpecs))
	}
}

func TestRequiresX86(t *testing.T) {
	// Known x86 instances
	if !RequiresX86("django__django-10087") {
		t.Error("django__django-10087 should require x86")
	}
	if !RequiresX86("matplotlib__matplotlib-13983") {
		t.Error("matplotlib__matplotlib-13983 should require x86")
	}
	// Non-x86 instance
	if RequiresX86("django__django-11099") {
		t.Error("django__django-11099 should not require x86")
	}
}
