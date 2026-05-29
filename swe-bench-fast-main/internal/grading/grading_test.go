package grading

import (
	"testing"

	"github.com/greynewell/swe-bench-fast/internal/parser"
)

func TestGradeFullResolved(t *testing.T) {
	statusMap := map[string]parser.TestStatus{
		"test_a": parser.StatusPassed,
		"test_b": parser.StatusPassed,
		"test_c": parser.StatusPassed,
	}
	f2p := []string{"test_a"}
	p2p := []string{"test_b", "test_c"}

	report := Grade(statusMap, f2p, p2p)

	if report.Resolved != ResolvedFull {
		t.Errorf("expected RESOLVED_FULL, got %s", report.Resolved)
	}
	if report.F2PPassed != 1 || report.F2PTotal != 1 {
		t.Errorf("expected F2P 1/1, got %d/%d", report.F2PPassed, report.F2PTotal)
	}
	if report.P2PPassed != 2 || report.P2PTotal != 2 {
		t.Errorf("expected P2P 2/2, got %d/%d", report.P2PPassed, report.P2PTotal)
	}
}

func TestGradePartialResolved(t *testing.T) {
	statusMap := map[string]parser.TestStatus{
		"test_a": parser.StatusPassed,
		"test_b": parser.StatusFailed,
		"test_c": parser.StatusPassed,
	}
	f2p := []string{"test_a", "test_b"}
	p2p := []string{"test_c"}

	report := Grade(statusMap, f2p, p2p)

	if report.Resolved != ResolvedPartial {
		t.Errorf("expected RESOLVED_PARTIAL, got %s", report.Resolved)
	}
	if report.F2PPassed != 1 {
		t.Errorf("expected F2P passed 1, got %d", report.F2PPassed)
	}
}

func TestGradeNotResolved(t *testing.T) {
	statusMap := map[string]parser.TestStatus{
		"test_a": parser.StatusFailed,
		"test_b": parser.StatusPassed,
	}
	f2p := []string{"test_a"}
	p2p := []string{"test_b"}

	report := Grade(statusMap, f2p, p2p)

	if report.Resolved != ResolvedNo {
		t.Errorf("expected RESOLVED_NO, got %s", report.Resolved)
	}
}

func TestGradeXFailCountsAsPass(t *testing.T) {
	statusMap := map[string]parser.TestStatus{
		"test_a": parser.StatusXFail,
		"test_b": parser.StatusPassed,
	}
	f2p := []string{"test_a"}
	p2p := []string{"test_b"}

	report := Grade(statusMap, f2p, p2p)

	if report.Resolved != ResolvedFull {
		t.Errorf("expected RESOLVED_FULL (XFAIL counts as pass), got %s", report.Resolved)
	}
}

func TestGradeMissingTestCountsAsFail(t *testing.T) {
	statusMap := map[string]parser.TestStatus{
		"test_b": parser.StatusPassed,
	}
	f2p := []string{"test_a"} // missing from map
	p2p := []string{"test_b"}

	report := Grade(statusMap, f2p, p2p)

	if report.Resolved != ResolvedNo {
		t.Errorf("expected RESOLVED_NO (missing test = fail), got %s", report.Resolved)
	}
	if report.F2PPassed != 0 {
		t.Errorf("expected 0 F2P passed, got %d", report.F2PPassed)
	}
}

func TestGradeP2PFailBreaksResolution(t *testing.T) {
	statusMap := map[string]parser.TestStatus{
		"test_a": parser.StatusPassed,
		"test_b": parser.StatusFailed,
	}
	f2p := []string{"test_a"}
	p2p := []string{"test_b"}

	report := Grade(statusMap, f2p, p2p)

	if report.Resolved != ResolvedNo {
		t.Errorf("expected RESOLVED_NO (P2P fail), got %s", report.Resolved)
	}
}

func TestGradeEmptyLists(t *testing.T) {
	statusMap := map[string]parser.TestStatus{}
	report := Grade(statusMap, nil, nil)

	if report.Resolved != ResolvedFull {
		t.Errorf("expected RESOLVED_FULL for empty lists, got %s", report.Resolved)
	}
}
