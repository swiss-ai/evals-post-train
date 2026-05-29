package parser

import (
	"testing"
)

func TestParseLogPytest(t *testing.T) {
	log := `
tests/test_foo.py::test_bar PASSED
tests/test_foo.py::test_baz FAILED
tests/test_foo.py::test_qux SKIPPED
tests/test_foo.py::test_err ERROR
tests/test_foo.py::test_xf XFAIL
`
	results := ParseLog("pytest-dev/pytest", log)

	tests := map[string]TestStatus{
		"tests/test_foo.py::test_bar": StatusPassed,
		"tests/test_foo.py::test_baz": StatusFailed,
		"tests/test_foo.py::test_qux": StatusSkipped,
		"tests/test_foo.py::test_err": StatusError,
		"tests/test_foo.py::test_xf":  StatusXFail,
	}

	for name, expected := range tests {
		got, ok := results[name]
		if !ok {
			t.Errorf("test %s not found in results", name)
			continue
		}
		if got != expected {
			t.Errorf("test %s: expected %s, got %s", name, expected, got)
		}
	}
}

func TestParseLogPytestV2ANSI(t *testing.T) {
	log := "\x1b[32mtests/test_foo.py::test_bar PASSED\x1b[0m\n" +
		"\x1b[31mtests/test_foo.py::test_baz FAILED\x1b[0m\n"

	results := ParseLog("astropy/astropy", log)

	if results["tests/test_foo.py::test_bar"] != StatusPassed {
		t.Errorf("expected PASSED, got %v", results["tests/test_foo.py::test_bar"])
	}
	if results["tests/test_foo.py::test_baz"] != StatusFailed {
		t.Errorf("expected FAILED, got %v", results["tests/test_foo.py::test_baz"])
	}
}

func TestParseLogDjango(t *testing.T) {
	log := `test_foo (admin_views.tests.AdminViewTest) ... ok
test_bar (admin_views.tests.AdminViewTest) ... FAIL
test_baz (admin_views.tests.AdminViewTest) ... ERROR
test_skip (admin_views.tests.AdminViewTest) ... skipped
`
	results := ParseLog("django/django", log)

	if results["admin_views.tests.AdminViewTest.test_foo"] != StatusPassed {
		t.Errorf("expected PASSED for test_foo, got %v", results["admin_views.tests.AdminViewTest.test_foo"])
	}
	if results["admin_views.tests.AdminViewTest.test_bar"] != StatusFailed {
		t.Errorf("expected FAILED for test_bar, got %v", results["admin_views.tests.AdminViewTest.test_bar"])
	}
	if results["admin_views.tests.AdminViewTest.test_baz"] != StatusError {
		t.Errorf("expected ERROR for test_baz, got %v", results["admin_views.tests.AdminViewTest.test_baz"])
	}
}

func TestParseLogDjangoDocstrings(t *testing.T) {
	// Django tests with docstring-based descriptions instead of test_* method names
	log := `test_foo (httpwrappers.tests.CookieTests) ... ok
Semicolons and commas are decoded. (httpwrappers.tests.CookieTests) ... ok
#13572 - QueryDict with a non-default encoding (httpwrappers.tests.QueryDictTests) ... ok
A copy of a QueryDict is mutable. (httpwrappers.tests.QueryDictTests) ... FAIL
Regression test for #8278: QueryDict.update(QueryDict) (httpwrappers.tests.QueryDictTests) ... ok
`
	results := ParseLog("django/django", log)

	// Standard test_* format should still work
	if results["httpwrappers.tests.CookieTests.test_foo"] != StatusPassed {
		t.Errorf("standard test_foo not found or wrong status: %v", results["httpwrappers.tests.CookieTests.test_foo"])
	}

	// Docstring-based test stored as altName: "description (class)"
	altName := "Semicolons and commas are decoded. (httpwrappers.tests.CookieTests)"
	if results[altName] != StatusPassed {
		t.Errorf("docstring test not found as altName %q: %v", altName, results[altName])
	}

	// Description with embedded parentheses
	altName2 := "Regression test for #8278: QueryDict.update(QueryDict) (httpwrappers.tests.QueryDictTests)"
	if results[altName2] != StatusPassed {
		t.Errorf("parenthesized docstring test not found: %v", results[altName2])
	}

	// Failed description-based test
	altName3 := "A copy of a QueryDict is mutable. (httpwrappers.tests.QueryDictTests)"
	if results[altName3] != StatusFailed {
		t.Errorf("failed docstring test: expected FAILED, got %v", results[altName3])
	}

	// Verify issue-reference format
	altName4 := "#13572 - QueryDict with a non-default encoding (httpwrappers.tests.QueryDictTests)"
	if results[altName4] != StatusPassed {
		t.Errorf("issue-ref docstring test not found: %v", results[altName4])
	}
}

func TestParseLogSympy(t *testing.T) {
	log := `test_foo ok
test_bar FAILED
test_baz SKIP
`
	results := ParseLog("sympy/sympy", log)

	if results["test_foo"] != StatusPassed {
		t.Errorf("expected PASSED, got %v", results["test_foo"])
	}
	if results["test_bar"] != StatusFailed {
		t.Errorf("expected FAILED, got %v", results["test_bar"])
	}
	if results["test_baz"] != StatusSkipped {
		t.Errorf("expected SKIPPED, got %v", results["test_baz"])
	}
}

func TestParseLogMatplotlib(t *testing.T) {
	log := "tests/test_foo.py::test_mouse[MouseButton.LEFT] PASSED\n" +
		"tests/test_foo.py::test_bar FAILED\n"

	results := ParseLog("matplotlib/matplotlib", log)

	if results["tests/test_foo.py::test_mouse[1]"] != StatusPassed {
		t.Errorf("expected PASSED with MouseButton replacement, got %v", results["tests/test_foo.py::test_mouse[1]"])
	}
}

func TestParseLogFallback(t *testing.T) {
	// Unknown repos fall back to pytest parser
	log := "tests/test_foo.py::test_bar PASSED\n"
	results := ParseLog("unknown/repo", log)

	if results["tests/test_foo.py::test_bar"] != StatusPassed {
		t.Errorf("expected fallback pytest parser to work, got %v", results["tests/test_foo.py::test_bar"])
	}
}
