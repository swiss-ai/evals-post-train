package dataset

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadDatasetJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "dataset.json")
	data := `[
		{
			"instance_id": "django__django-11099",
			"repo": "django/django",
			"base_commit": "abc123",
			"patch": "diff --git a/foo b/foo",
			"test_patch": "diff --git a/tests/foo b/tests/foo",
			"FAIL_TO_PASS": ["test_foo"],
			"PASS_TO_PASS": ["test_bar"],
			"version": "3.0"
		}
	]`
	if err := os.WriteFile(path, []byte(data), 0644); err != nil {
		t.Fatal(err)
	}

	instances, err := LoadDataset(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(instances) != 1 {
		t.Fatalf("expected 1 instance, got %d", len(instances))
	}
	if instances[0].InstanceID != "django__django-11099" {
		t.Errorf("expected instance_id django__django-11099, got %s", instances[0].InstanceID)
	}
	if instances[0].Repo != "django/django" {
		t.Errorf("expected repo django/django, got %s", instances[0].Repo)
	}
	if len(instances[0].FailToPass) != 1 || instances[0].FailToPass[0] != "test_foo" {
		t.Errorf("unexpected FAIL_TO_PASS: %v", instances[0].FailToPass)
	}
}

func TestLoadDatasetJSONL(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "dataset.jsonl")
	data := `{"instance_id": "pytest__pytest-1234", "repo": "pytest-dev/pytest", "base_commit": "def456", "patch": "", "test_patch": "", "FAIL_TO_PASS": [], "PASS_TO_PASS": [], "version": "5.0"}
{"instance_id": "flask__flask-5678", "repo": "flask/flask", "base_commit": "ghi789", "patch": "", "test_patch": "", "FAIL_TO_PASS": [], "PASS_TO_PASS": [], "version": "2.0"}
`
	if err := os.WriteFile(path, []byte(data), 0644); err != nil {
		t.Fatal(err)
	}

	instances, err := LoadDataset(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(instances) != 2 {
		t.Fatalf("expected 2 instances, got %d", len(instances))
	}
}

func TestLoadPredictions(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "preds.jsonl")
	data := `{"instance_id": "django__django-11099", "model_name_or_path": "gpt-4", "model_patch": "--- a/foo\n+++ b/foo\n"}
{"instance_id": "pytest__pytest-1234", "model_name_or_path": "gpt-4", "model_patch": "--- a/bar\n+++ b/bar\n"}
`
	if err := os.WriteFile(path, []byte(data), 0644); err != nil {
		t.Fatal(err)
	}

	preds, err := LoadPredictions(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(preds) != 2 {
		t.Fatalf("expected 2 predictions, got %d", len(preds))
	}
	if preds[0].ModelPatch == "" {
		t.Error("expected non-empty model_patch")
	}
}

func TestMerge(t *testing.T) {
	instances := []Instance{
		{InstanceID: "a"},
		{InstanceID: "b"},
		{InstanceID: "c"},
	}
	predictions := []Prediction{
		{InstanceID: "a", ModelPatch: "patch-a"},
		{InstanceID: "c", ModelPatch: "patch-c"},
		{InstanceID: "d", ModelPatch: "patch-d"}, // no matching instance
	}

	tasks := Merge(instances, predictions)
	if len(tasks) != 2 {
		t.Fatalf("expected 2 merged tasks, got %d", len(tasks))
	}
	if tasks[0].Instance.InstanceID != "a" || tasks[0].Prediction.ModelPatch != "patch-a" {
		t.Errorf("unexpected first task: %+v", tasks[0])
	}
	if tasks[1].Instance.InstanceID != "c" {
		t.Errorf("unexpected second task: %+v", tasks[1])
	}
}
