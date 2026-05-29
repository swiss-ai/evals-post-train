package eval

import (
	"strings"
	"testing"

	"github.com/greynewell/swe-bench-fast/internal/dataset"
	"github.com/greynewell/swe-bench-fast/internal/spec"
)

func TestTestDirectivesFromPatch(t *testing.T) {
	patch := `diff --git a/tests/admin_views/tests.py b/tests/admin_views/tests.py
index abc..def 100644
--- a/tests/admin_views/tests.py
+++ b/tests/admin_views/tests.py
@@ -1 +1 @@
-old
+new
`
	// For Django, should extract module name
	directives := TestDirectivesFromPatch("django/django", patch)
	if len(directives) != 1 || directives[0] != "admin_views" {
		t.Errorf("django directive: expected [admin_views], got %v", directives)
	}

	// For non-Django repos, should extract file paths
	directives = TestDirectivesFromPatch("pytest-dev/pytest", patch)
	if len(directives) != 1 || directives[0] != "tests/admin_views/tests.py" {
		t.Errorf("pytest directive: expected [tests/admin_views/tests.py], got %v", directives)
	}
}

func TestGenerateEvalScript(t *testing.T) {
	inst := dataset.Instance{
		InstanceID: "django__django-11099",
		Repo:       "django/django",
		BaseCommit: "abc123def",
		TestPatch: `diff --git a/tests/admin_views/tests.py b/tests/admin_views/tests.py
--- a/tests/admin_views/tests.py
+++ b/tests/admin_views/tests.py
@@ -1 +1 @@
-old
+new
`,
		Version: "3.0",
	}

	rs := spec.RepoSpec{
		Python:  "3.6",
		Install: "python -m pip install -e .",
		TestCmd: "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
	}

	script := GenerateEvalScript(inst, rs)

	// Check essential components
	if !strings.Contains(script, "#!/bin/bash") {
		t.Error("missing shebang")
	}
	if !strings.Contains(script, "set -xo pipefail") {
		t.Error("missing set flags")
	}
	if !strings.Contains(script, "conda activate testbed") {
		t.Error("missing conda activate")
	}
	if !strings.Contains(script, "git checkout abc123def") {
		t.Error("missing git checkout")
	}
	if !strings.Contains(script, "EOF_114329324912") {
		t.Error("missing heredoc delimiter")
	}
	if !strings.Contains(script, ">>>>> Start Test Output") {
		t.Error("missing test output markers")
	}
	if !strings.Contains(script, "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1") {
		t.Error("missing Django test command")
	}
	if !strings.Contains(script, "admin_views") {
		t.Error("missing Django test directive")
	}
}
