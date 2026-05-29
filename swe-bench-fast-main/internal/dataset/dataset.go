package dataset

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// StringList is a []string that can be unmarshalled from either a JSON array
// or a JSON string containing a JSON array (as in the SWE-bench HuggingFace format).
type StringList []string

func (s *StringList) UnmarshalJSON(data []byte) error {
	// Try as a regular string array first
	var arr []string
	if err := json.Unmarshal(data, &arr); err == nil {
		*s = arr
		return nil
	}
	// Try as a JSON-encoded string containing an array
	var str string
	if err := json.Unmarshal(data, &str); err != nil {
		return fmt.Errorf("StringList: expected array or string, got %s", string(data[:min(len(data), 50)]))
	}
	if err := json.Unmarshal([]byte(str), &arr); err != nil {
		return fmt.Errorf("StringList: cannot parse string as array: %w", err)
	}
	*s = arr
	return nil
}

// Instance represents a single SWE-bench evaluation instance.
type Instance struct {
	InstanceID string     `json:"instance_id"`
	Repo       string     `json:"repo"`
	BaseCommit string     `json:"base_commit"`
	Patch      string     `json:"patch"`
	TestPatch  string     `json:"test_patch"`
	FailToPass StringList `json:"FAIL_TO_PASS"`
	PassToPass StringList `json:"PASS_TO_PASS"`
	Version    string     `json:"version"`
}

// Prediction represents a model's predicted patch for an instance.
type Prediction struct {
	InstanceID string `json:"instance_id"`
	ModelName  string `json:"model_name_or_path"`
	ModelPatch string `json:"model_patch"`
}

// MergeTask combines an Instance with its corresponding Prediction for evaluation.
type MergeTask struct {
	Instance   Instance
	Prediction Prediction
}

// LoadDataset reads SWE-bench instances from a JSON array or JSONL file.
func LoadDataset(path string) ([]Instance, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read dataset: %w", err)
	}

	trimmed := strings.TrimSpace(string(data))

	// Try JSON array first
	if strings.HasPrefix(trimmed, "[") {
		var instances []Instance
		if err := json.Unmarshal(data, &instances); err != nil {
			return nil, fmt.Errorf("parse dataset JSON array: %w", err)
		}
		return instances, nil
	}

	// Fall back to JSONL
	return parseJSONL[Instance](trimmed)
}

// LoadPredictions reads model predictions from a JSONL file.
func LoadPredictions(path string) ([]Prediction, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read predictions: %w", err)
	}
	return parseJSONL[Prediction](strings.TrimSpace(string(data)))
}

func parseJSONL[T any](data string) ([]T, error) {
	var results []T
	scanner := bufio.NewScanner(strings.NewReader(data))
	scanner.Buffer(make([]byte, 0, 64*1024), 10*1024*1024) // 10MB max line
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var item T
		if err := json.Unmarshal([]byte(line), &item); err != nil {
			return nil, fmt.Errorf("parse line %d: %w", lineNum, err)
		}
		results = append(results, item)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan JSONL: %w", err)
	}
	return results, nil
}

// Merge performs an inner join of instances and predictions on instance_id.
// Only instances that have a corresponding prediction are returned.
func Merge(instances []Instance, predictions []Prediction) []MergeTask {
	instMap := make(map[string]Instance, len(instances))
	for _, inst := range instances {
		instMap[inst.InstanceID] = inst
	}

	var tasks []MergeTask
	for _, pred := range predictions {
		if inst, ok := instMap[pred.InstanceID]; ok {
			tasks = append(tasks, MergeTask{
				Instance:   inst,
				Prediction: pred,
			})
		}
	}
	return tasks
}
