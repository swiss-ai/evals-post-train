package builder

import (
	"archive/tar"
	"io"
	"testing"

	"github.com/greynewell/swe-bench-fast/internal/spec"
)

func TestMakeBuildContext(t *testing.T) {
	files := map[string][]byte{
		"Dockerfile":   []byte("FROM ubuntu:22.04\n"),
		"setup_env.sh": []byte("#!/bin/bash\necho hello\n"),
	}

	reader := makeBuildContext(files)
	tr := tar.NewReader(reader)

	found := make(map[string]bool)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("tar.Next() error: %v", err)
		}
		found[hdr.Name] = true

		content, err := io.ReadAll(tr)
		if err != nil {
			t.Fatalf("read tar entry %s: %v", hdr.Name, err)
		}

		expected, ok := files[hdr.Name]
		if !ok {
			t.Errorf("unexpected file in tar: %s", hdr.Name)
			continue
		}
		if string(content) != string(expected) {
			t.Errorf("file %s content = %q, want %q", hdr.Name, string(content), string(expected))
		}
	}

	for name := range files {
		if !found[name] {
			t.Errorf("expected file %s not found in tar", name)
		}
	}
}

func TestDedupStrings(t *testing.T) {
	specs := []spec.ImageSpec{
		{BaseTag: "base:a", EnvTag: "env:1"},
		{BaseTag: "base:a", EnvTag: "env:2"},
		{BaseTag: "base:b", EnvTag: "env:1"},
		{BaseTag: "base:b", EnvTag: "env:3"},
	}

	baseTags := dedupStrings(specs, func(s spec.ImageSpec) string { return s.BaseTag })
	if len(baseTags) != 2 {
		t.Errorf("expected 2 unique base tags, got %d: %v", len(baseTags), baseTags)
	}

	envTags := dedupStrings(specs, func(s spec.ImageSpec) string { return s.EnvTag })
	if len(envTags) != 3 {
		t.Errorf("expected 3 unique env tags, got %d: %v", len(envTags), envTags)
	}
}
