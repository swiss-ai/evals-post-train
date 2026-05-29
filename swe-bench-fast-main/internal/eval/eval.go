package eval

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/greynewell/swe-bench-fast/internal/dataset"
	"github.com/greynewell/swe-bench-fast/internal/spec"
)

// testFileRe matches test file paths from unified diff headers.
var testFileRe = regexp.MustCompile(`(?m)^diff --git a/(.+?) b/(.+?)$`)

// TestDirectivesFromPatch extracts test file paths from a unified diff patch.
func TestDirectivesFromPatch(repo, testPatch string) []string {
	matches := testFileRe.FindAllStringSubmatch(testPatch, -1)
	seen := make(map[string]bool)
	var files []string
	for _, m := range matches {
		path := m[2] // b/ side
		if seen[path] {
			continue
		}
		seen[path] = true

		// Django uses module-style paths for its test runner
		if strings.HasPrefix(repo, "django/") {
			directive := djangoTestDirective(path)
			if directive != "" {
				files = append(files, directive)
			}
			continue
		}

		files = append(files, path)
	}
	return files
}

// djangoTestDirective converts a file path to a Django test module directive.
// e.g., tests/admin_views/tests.py → admin_views
func djangoTestDirective(path string) string {
	if !strings.HasPrefix(path, "tests/") {
		return ""
	}
	// Remove "tests/" prefix
	rest := strings.TrimPrefix(path, "tests/")
	// Take just the first directory component
	parts := strings.SplitN(rest, "/", 2)
	if len(parts) == 0 {
		return ""
	}
	return parts[0]
}

// TestFilesFromPatch extracts the b-side file paths from a unified diff.
func TestFilesFromPatch(testPatch string) []string {
	matches := testFileRe.FindAllStringSubmatch(testPatch, -1)
	seen := make(map[string]bool)
	var files []string
	for _, m := range matches {
		path := m[2]
		if !seen[path] {
			seen[path] = true
			files = append(files, path)
		}
	}
	return files
}

// GenerateEvalScript produces the bash eval script for a given instance.
// This matches the SWE-bench script format exactly.
func GenerateEvalScript(inst dataset.Instance, rs spec.RepoSpec) string {
	var b strings.Builder
	b.WriteString("#!/bin/bash\n")
	b.WriteString("set -xo pipefail\n")
	b.WriteString(GenerateEvalScriptBody(inst, rs))
	return b.String()
}

// GenerateEvalScriptBody returns the eval script body without the shebang/header.
// This can be embedded into compound scripts.
func GenerateEvalScriptBody(inst dataset.Instance, rs spec.RepoSpec) string {
	directives := TestDirectivesFromPatch(inst.Repo, inst.TestPatch)
	testFiles := TestFilesFromPatch(inst.TestPatch)

	var b strings.Builder
	b.WriteString("source /opt/miniconda3/bin/activate\n")
	if !rs.NoUseEnv {
		b.WriteString("conda activate testbed\n")
	}
	b.WriteString("cd /testbed\n")

	// Emit eval commands (locale setup, env vars, etc.)
	for _, cmd := range rs.EvalEnvVars {
		b.WriteString(cmd + "\n")
	}

	// Safe directory + diagnostics — matches upstream
	b.WriteString("git config --global --add safe.directory /testbed\n")
	b.WriteString("cd /testbed\n")
	b.WriteString("git status\n")
	b.WriteString("git show\n")
	b.WriteString(fmt.Sprintf("git -c core.fileMode=false diff %s\n", inst.BaseCommit))
	b.WriteString("source /opt/miniconda3/bin/activate\n")
	if !rs.NoUseEnv {
		b.WriteString("conda activate testbed\n")
	}

	// Re-run install command — matches upstream eval behavior
	if rs.Install != "" {
		b.WriteString(rs.Install + "\n")
	}

	// Non-root user setup for old matplotlib etc.
	if rs.ExecAsNonRoot {
		b.WriteString("useradd -m nonroot 2>/dev/null || true\n")
		b.WriteString("chown -R nonroot:nonroot /testbed\n")
	}

	// Checkout test files to base commit state before applying test patch
	if len(testFiles) > 0 {
		b.WriteString(fmt.Sprintf("git checkout %s -- %s\n", inst.BaseCommit, strings.Join(testFiles, " ")))
	}

	// Apply the test patch
	b.WriteString("git apply -v - <<'EOF_114329324912'\n")
	b.WriteString(inst.TestPatch)
	if !strings.HasSuffix(inst.TestPatch, "\n") {
		b.WriteString("\n")
	}
	b.WriteString("EOF_114329324912\n")

	// Run tests
	b.WriteString(": '>>>>> Start Test Output'\n")
	if rs.ExecAsNonRoot {
		b.WriteString("runuser -u nonroot -- ")
	}
	b.WriteString(rs.TestCmd)
	if len(directives) > 0 {
		b.WriteString(" " + strings.Join(directives, " "))
	}
	b.WriteString("\n")
	b.WriteString(": '>>>>> End Test Output'\n")

	// Restore test files to base commit state
	if len(testFiles) > 0 {
		b.WriteString(fmt.Sprintf("git checkout %s -- %s\n", inst.BaseCommit, strings.Join(testFiles, " ")))
	}

	return b.String()
}
