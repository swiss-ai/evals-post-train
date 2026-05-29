package docker

import (
	"context"
	"fmt"
	"sync"
	"time"
)

const DefaultMemLimit = "4g"

var (
	defaultClient *Client
	clientOnce    sync.Once
	clientErr     error
)

// getClient returns a lazily-initialized default Docker SDK client.
func getClient() (*Client, error) {
	clientOnce.Do(func() {
		defaultClient, clientErr = NewClient()
	})
	return defaultClient, clientErr
}

// SetDefaultClient sets the default client used by package-level functions.
// Must be called before any other package-level function.
func SetDefaultClient(c *Client) {
	defaultClient = c
	clientErr = nil
}

// Pull fetches a Docker image if not already present.
func Pull(ctx context.Context, image string) error {
	c, err := getClient()
	if err != nil {
		return fmt.Errorf("docker client: %w", err)
	}
	return c.ImagePull(ctx, image)
}

// ImageExists checks whether an image exists locally.
func ImageExists(ctx context.Context, ref string) (bool, error) {
	c, err := getClient()
	if err != nil {
		return false, fmt.Errorf("docker client: %w", err)
	}
	return c.ImageExists(ctx, ref)
}

// Create creates a new container from the given image and returns its ID.
func Create(ctx context.Context, image, name, memLimit string) (string, error) {
	c, err := getClient()
	if err != nil {
		return "", fmt.Errorf("docker client: %w", err)
	}

	if memLimit == "" {
		memLimit = DefaultMemLimit
	}

	mem, err := ParseMemLimit(memLimit)
	if err != nil {
		return "", fmt.Errorf("parse mem limit: %w", err)
	}

	return c.ContainerCreate(ctx, ContainerOpts{
		Name:     name,
		Image:    image,
		MemLimit: mem,
		Network:  "none",
	})
}

// CreateWithOpts creates a container with full ContainerOpts.
func CreateWithOpts(ctx context.Context, opts ContainerOpts) (string, error) {
	c, err := getClient()
	if err != nil {
		return "", fmt.Errorf("docker client: %w", err)
	}
	return c.ContainerCreate(ctx, opts)
}

// Start starts a stopped container.
func Start(ctx context.Context, id string) error {
	c, err := getClient()
	if err != nil {
		return fmt.Errorf("docker client: %w", err)
	}
	return c.ContainerStart(ctx, id)
}

// Exec runs a command inside a running container and returns stdout, stderr, and exit code.
func Exec(ctx context.Context, id, command string, timeout time.Duration) (stdout, stderr string, exitCode int, err error) {
	c, cErr := getClient()
	if cErr != nil {
		return "", "", -1, fmt.Errorf("docker client: %w", cErr)
	}

	result, err := c.ContainerExec(ctx, id, []string{"/bin/bash", "-c", command}, timeout)
	if err != nil {
		return "", "", -1, err
	}
	return result.Stdout, result.Stderr, result.ExitCode, nil
}

// CopyTo writes content to a file inside the container.
func CopyTo(ctx context.Context, id, content, destPath string) error {
	c, err := getClient()
	if err != nil {
		return fmt.Errorf("docker client: %w", err)
	}
	return c.CopyToContainer(ctx, id, []byte(content), destPath)
}

// Remove force-removes a container.
func Remove(ctx context.Context, id string) error {
	c, err := getClient()
	if err != nil {
		return fmt.Errorf("docker client: %w", err)
	}
	return c.ContainerRemove(ctx, id)
}

// ImageName returns the Epoch Research pre-built image name for an instance.
func ImageName(registry, prefix, instanceID, arch string) string {
	return fmt.Sprintf("%s/%s.%s.%s:latest", registry, prefix, arch, instanceID)
}
