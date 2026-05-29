package grading

import (
	"strings"

	"github.com/greynewell/swe-bench-fast/internal/parser"
)

// ResolvedStatus represents the resolution outcome of an evaluation.
type ResolvedStatus string

const (
	ResolvedFull    ResolvedStatus = "RESOLVED_FULL"
	ResolvedPartial ResolvedStatus = "RESOLVED_PARTIAL"
	ResolvedNo      ResolvedStatus = "RESOLVED_NO"
)

// Report contains the full evaluation result for a single instance.
type Report struct {
	InstanceID   string         `json:"instance_id"`
	Resolved     ResolvedStatus `json:"resolved"`
	F2PTotal     int            `json:"f2p_total"`
	F2PPassed    int            `json:"f2p_passed"`
	P2PTotal     int            `json:"p2p_total"`
	P2PPassed    int            `json:"p2p_passed"`
	PatchApplied bool           `json:"patch_applied"`
	Error        string         `json:"error,omitempty"`
	DurationMS   int64          `json:"duration_ms"`
}

// testPasses returns true if the test status indicates a pass.
func testPasses(status parser.TestStatus) bool {
	return status == parser.StatusPassed || status == parser.StatusXFail
}

// lookupStatus finds a test name in the status map. It first tries an exact match,
// then falls back to prefix matching for SWE-bench dataset entries that are truncated
// at spaces (e.g., parametrized test names like "test_foo[Type-*1" instead of
// "test_foo[Type-*1 xfailed*]").
func lookupStatus(statusMap map[string]parser.TestStatus, test string) (parser.TestStatus, bool) {
	if status, ok := statusMap[test]; ok {
		return status, true
	}
	// Prefix match: the dataset entry may be truncated at a space
	for name, status := range statusMap {
		if strings.HasPrefix(name, test+" ") {
			return status, true
		}
	}
	return "", false
}

// Grade evaluates the test results against the expected FAIL_TO_PASS and PASS_TO_PASS lists.
// A test is considered passing if its status is PASSED or XFAIL.
// A test is considered failing if its status is FAILED, ERROR, or it's missing from the status map.
func Grade(statusMap map[string]parser.TestStatus, f2p, p2p []string) Report {
	report := Report{
		F2PTotal: len(f2p),
		P2PTotal: len(p2p),
	}

	f2pAllPass := true
	f2pAnyPass := false
	for _, test := range f2p {
		status, ok := lookupStatus(statusMap, test)
		if ok && testPasses(status) {
			report.F2PPassed++
			f2pAnyPass = true
		} else {
			f2pAllPass = false
		}
	}

	p2pAllPass := true
	for _, test := range p2p {
		status, ok := lookupStatus(statusMap, test)
		if ok && testPasses(status) {
			report.P2PPassed++
		} else {
			p2pAllPass = false
		}
	}

	switch {
	case f2pAllPass && p2pAllPass:
		report.Resolved = ResolvedFull
	case f2pAnyPass && p2pAllPass:
		report.Resolved = ResolvedPartial
	default:
		report.Resolved = ResolvedNo
	}

	return report
}
