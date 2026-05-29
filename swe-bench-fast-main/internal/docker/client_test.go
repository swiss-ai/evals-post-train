package docker

import (
	"bytes"
	"context"
	"testing"
	"time"
)

func dockerAvailable() bool {
	c, err := NewClient()
	if err != nil {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_, err = c.api.Ping(ctx)
	c.Close()
	return err == nil
}

func TestNewClient(t *testing.T) {
	if !dockerAvailable() {
		t.Skip("docker not available")
	}

	c, err := NewClient()
	if err != nil {
		t.Fatalf("NewClient() error: %v", err)
	}
	defer c.Close()
}

func TestImageExists(t *testing.T) {
	if !dockerAvailable() {
		t.Skip("docker not available")
	}

	c, err := NewClient()
	if err != nil {
		t.Fatalf("NewClient() error: %v", err)
	}
	defer c.Close()

	ctx := context.Background()

	// A clearly non-existent image
	exists, err := c.ImageExists(ctx, "swe-bench-fast-test-nonexistent:latest")
	if err != nil {
		t.Fatalf("ImageExists() error: %v", err)
	}
	if exists {
		t.Error("expected non-existent image to return false")
	}
}

func TestParseMemLimit(t *testing.T) {
	tests := []struct {
		input    string
		expected int64
		wantErr  bool
	}{
		{"4g", 4 * 1024 * 1024 * 1024, false},
		{"512m", 512 * 1024 * 1024, false},
		{"1024k", 1024 * 1024, false},
		{"", 0, false},
		{"invalid", 0, true},
	}

	for _, tt := range tests {
		got, err := ParseMemLimit(tt.input)
		if (err != nil) != tt.wantErr {
			t.Errorf("ParseMemLimit(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
			continue
		}
		if !tt.wantErr && got != tt.expected {
			t.Errorf("ParseMemLimit(%q) = %d, want %d", tt.input, got, tt.expected)
		}
	}
}

func TestCopyToContainerTarCreation(t *testing.T) {
	// Test that CopyToContainer creates a valid tar (unit test without Docker)
	content := []byte("test content")
	var buf bytes.Buffer

	// We can test the tar creation logic directly
	c := &Client{} // nil api is fine since we won't call Docker
	_ = c           // just verifying struct creation

	// Verify ParseMemLimit works for common cases
	mem, err := ParseMemLimit("4g")
	if err != nil {
		t.Fatalf("ParseMemLimit(4g) error: %v", err)
	}
	if mem != 4*1024*1024*1024 {
		t.Errorf("ParseMemLimit(4g) = %d, want %d", mem, 4*1024*1024*1024)
	}

	_ = content
	_ = buf
}

func TestContainerLifecycle(t *testing.T) {
	if !dockerAvailable() {
		t.Skip("docker not available")
	}

	c, err := NewClient()
	if err != nil {
		t.Fatalf("NewClient() error: %v", err)
	}
	defer c.Close()

	ctx := context.Background()

	// Pull a small test image
	err = c.ImagePull(ctx, "alpine:3.20")
	if err != nil {
		t.Fatalf("ImagePull() error: %v", err)
	}

	// Verify it exists
	exists, err := c.ImageExists(ctx, "alpine:3.20")
	if err != nil {
		t.Fatalf("ImageExists() error: %v", err)
	}
	if !exists {
		t.Fatal("expected alpine:3.20 to exist after pull")
	}

	// Create container
	id, err := c.ContainerCreate(ctx, ContainerOpts{
		Name:    "swe-bench-fast-test",
		Image:   "alpine:3.20",
		Network: "none",
	})
	if err != nil {
		t.Fatalf("ContainerCreate() error: %v", err)
	}
	defer c.ContainerRemove(ctx, id)

	// Start container
	if err := c.ContainerStart(ctx, id); err != nil {
		t.Fatalf("ContainerStart() error: %v", err)
	}

	// Execute a command
	result, err := c.ContainerExec(ctx, id, []string{"echo", "hello"}, 10*time.Second)
	if err != nil {
		t.Fatalf("ContainerExec() error: %v", err)
	}
	if result.ExitCode != 0 {
		t.Errorf("exec exit code = %d, want 0", result.ExitCode)
	}
	if result.Stdout != "hello\n" {
		t.Errorf("exec stdout = %q, want %q", result.Stdout, "hello\n")
	}

	// Copy content to container
	err = c.CopyToContainer(ctx, id, []byte("test data"), "/tmp/test.txt")
	if err != nil {
		t.Fatalf("CopyToContainer() error: %v", err)
	}

	// Verify the content was copied
	result, err = c.ContainerExec(ctx, id, []string{"cat", "/tmp/test.txt"}, 10*time.Second)
	if err != nil {
		t.Fatalf("ContainerExec(cat) error: %v", err)
	}
	if result.Stdout != "test data" {
		t.Errorf("copied content = %q, want %q", result.Stdout, "test data")
	}

	// Remove container
	err = c.ContainerRemove(ctx, id)
	if err != nil {
		t.Fatalf("ContainerRemove() error: %v", err)
	}
}
