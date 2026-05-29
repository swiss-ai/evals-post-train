package parser

import (
	"regexp"
	"strings"
)

// TestStatus represents the outcome of a single test case.
type TestStatus string

const (
	StatusPassed  TestStatus = "PASSED"
	StatusFailed  TestStatus = "FAILED"
	StatusSkipped TestStatus = "SKIPPED"
	StatusError   TestStatus = "ERROR"
	StatusXFail   TestStatus = "XFAIL"
)

// parserFunc is the signature for repo-specific log parsers.
type parserFunc = func(string) map[string]TestStatus

// repoParserMap maps repo names to their parser functions.
var repoParserMap = map[string]parserFunc{
	"pytest-dev/pytest":            parseLogPytest,
	"flask/flask":                  parseLogPytest,
	"pallets/flask":                parseLogPytest,
	"marshmallow-code/marshmallow": parseLogPytest,
	"pydata/xarray":                parseLogPytest,
	"astropy/astropy":              parseLogPytestV2,
	"scikit-learn/scikit-learn":    parseLogPytestV2,
	"sphinx-doc/sphinx":            parseLogPytestV2,
	"psf/requests":                 parseLogPytestOptions,
	"pydicom/pydicom":              parseLogPytestOptions,
	"pylint-dev/pylint":            parseLogPytestOptions,
	"django/django":                parseLogDjango,
	"sympy/sympy":                  parseLogSympy,
	"matplotlib/matplotlib":        parseLogMatplotlib,
	"mwaskom/seaborn":              parseLogSeaborn,
}

// ParseLog dispatches to the correct parser for the given repo.
// If no specific parser exists, falls back to the generic pytest parser.
func ParseLog(repo, log string) map[string]TestStatus {
	fn, ok := repoParserMap[repo]
	if !ok {
		fn = parseLogPytest
	}
	return fn(log)
}

// --- Pytest parser (standard) ---

var pytestResultRe = regexp.MustCompile(`(?m)^(PASSED|FAILED|ERROR|SKIPPED|XFAIL)\s+(.+)$`)
var pytestShortRe = regexp.MustCompile(`(?m)^(.+?)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL)\s*$`)

func parseLogPytest(log string) map[string]TestStatus {
	results := make(map[string]TestStatus)

	// Match "STATUS test::name" format
	for _, m := range pytestResultRe.FindAllStringSubmatch(log, -1) {
		status := TestStatus(m[1])
		testName := strings.TrimSpace(m[2])
		results[testName] = status
	}

	// Match "test::name STATUS" format (more common in pytest output)
	for _, m := range pytestShortRe.FindAllStringSubmatch(log, -1) {
		testName := strings.TrimSpace(m[1])
		status := TestStatus(m[2])
		// Don't overwrite if already found
		if _, exists := results[testName]; !exists {
			results[testName] = status
		}
	}

	return results
}

// --- Pytest V2 parser (astropy, scikit-learn, sphinx) ---
// Handles ANSI escape codes and more verbose output formats.

var ansiRe = regexp.MustCompile(`\x1b\[[0-9;]*m`)

func stripANSI(s string) string {
	return ansiRe.ReplaceAllString(s, "")
}

func parseLogPytestV2(log string) map[string]TestStatus {
	cleaned := stripANSI(log)
	return parseLogPytest(cleaned)
}

// --- Pytest Options parser (pydicom, requests, pylint) ---
// Handles bracket-style parametrized test names and normalizes them.

var bracketParamRe = regexp.MustCompile(`\[.*?\]`)

func parseLogPytestOptions(log string) map[string]TestStatus {
	cleaned := stripANSI(log)
	results := make(map[string]TestStatus)

	for _, m := range pytestShortRe.FindAllStringSubmatch(cleaned, -1) {
		testName := strings.TrimSpace(m[1])
		status := TestStatus(m[2])
		results[testName] = status
	}

	for _, m := range pytestResultRe.FindAllStringSubmatch(cleaned, -1) {
		status := TestStatus(m[1])
		testName := strings.TrimSpace(m[2])
		if _, exists := results[testName]; !exists {
			results[testName] = status
		}
	}

	// Also store normalized (bracket-stripped) versions
	normalized := make(map[string]TestStatus)
	for k, v := range results {
		normalized[k] = v
		stripped := bracketParamRe.ReplaceAllString(k, "")
		if stripped != k {
			if _, exists := normalized[stripped]; !exists {
				normalized[stripped] = v
			}
		}
	}

	return normalized
}

// --- Django parser ---

// djangoSingleRe matches single-line: "test_name (module.Class) ... ok"
var djangoSingleRe = regexp.MustCompile(`(?m)^(.+?)\s+\((\S+)\)\s+\.\.\.\s+(ok|FAIL|ERROR|skipped)\s*$`)

// djangoIdentRe matches the identifier line of a two-line pair: "test_name (module.Class)"
var djangoIdentRe = regexp.MustCompile(`^(\S+)\s+\((\S+)\)\s*$`)

// djangoDocRe matches the docstring+status line: "Some description. ... ok"
var djangoDocRe = regexp.MustCompile(`^.+\.\.\.\s+(ok|FAIL|ERROR|skipped)\s*$`)

func parseLogDjango(log string) map[string]TestStatus {
	results := make(map[string]TestStatus)

	// First pass: single-line format "test_name (module.Class) ... ok"
	for _, m := range djangoSingleRe.FindAllStringSubmatch(log, -1) {
		testMethod := m[1]
		testClass := m[2]
		status := djangoStatus(m[3])
		fullName := testClass + "." + testMethod
		results[fullName] = status
		altName := testMethod + " (" + testClass + ")"
		results[altName] = status
	}

	// Second pass: two-line format where docstring tests split across lines:
	//   test_name (module.Class)
	//   Docstring text. ... ok
	lines := strings.Split(log, "\n")
	for i := 0; i < len(lines)-1; i++ {
		m := djangoIdentRe.FindStringSubmatch(strings.TrimSpace(lines[i]))
		if m == nil {
			continue
		}
		nextLine := strings.TrimSpace(lines[i+1])
		dm := djangoDocRe.FindStringSubmatch(nextLine)
		if dm == nil {
			continue
		}
		testMethod := m[1]
		testClass := m[2]
		status := djangoStatus(dm[1])
		fullName := testClass + "." + testMethod
		if _, exists := results[fullName]; !exists {
			results[fullName] = status
			altName := testMethod + " (" + testClass + ")"
			results[altName] = status
		}
	}

	return results
}

func djangoStatus(s string) TestStatus {
	switch strings.ToLower(s) {
	case "ok":
		return StatusPassed
	case "fail":
		return StatusFailed
	case "error":
		return StatusError
	case "skipped":
		return StatusSkipped
	default:
		return StatusFailed
	}
}

// --- Sympy parser ---

var sympyTestRe = regexp.MustCompile(`(?m)^(test_\S+)\s+(\.\.\.\s+)?(ok|FAILED|SKIP|ERROR|E|F|f)\s*$`)
var sympyResultLineRe = regexp.MustCompile(`(?m)^(\S+)\s+(ok|FAILED|SKIP|ERROR)\s*$`)

func parseLogSympy(log string) map[string]TestStatus {
	results := make(map[string]TestStatus)

	for _, m := range sympyTestRe.FindAllStringSubmatch(log, -1) {
		testName := m[1]
		status := sympyStatus(m[3])
		results[testName] = status
	}

	for _, m := range sympyResultLineRe.FindAllStringSubmatch(log, -1) {
		testName := m[1]
		status := sympyStatus(m[2])
		if _, exists := results[testName]; !exists {
			results[testName] = status
		}
	}

	return results
}

func sympyStatus(s string) TestStatus {
	switch s {
	case "ok":
		return StatusPassed
	case "FAILED", "F", "f":
		return StatusFailed
	case "SKIP":
		return StatusSkipped
	case "ERROR", "E":
		return StatusError
	default:
		return StatusFailed
	}
}

// --- Matplotlib parser ---
// Replaces MouseButton references and delegates to pytest.

func parseLogMatplotlib(log string) map[string]TestStatus {
	// Some matplotlib logs contain MouseButton references that break parsing
	cleaned := strings.ReplaceAll(log, "MouseButton.LEFT", "1")
	cleaned = strings.ReplaceAll(cleaned, "MouseButton.RIGHT", "3")
	cleaned = strings.ReplaceAll(cleaned, "MouseButton.MIDDLE", "2")
	cleaned = stripANSI(cleaned)
	return parseLogPytest(cleaned)
}

// --- Seaborn parser ---

func parseLogSeaborn(log string) map[string]TestStatus {
	cleaned := stripANSI(log)
	return parseLogPytest(cleaned)
}
