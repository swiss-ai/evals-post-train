package docker

import (
	"archive/tar"
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/client"
	"github.com/docker/docker/pkg/stdcopy"
	"github.com/docker/go-units"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// Client wraps the Docker Go SDK client for all container operations.
type Client struct {
	api  *client.Client
	host string // resolved DOCKER_HOST (e.g. "unix:///..."), empty = default
}

// ContainerOpts holds options for creating a container.
type ContainerOpts struct {
	Name     string
	Image    string
	Cmd      []string          // override CMD; defaults to ["sleep", "infinity"]
	MemLimit int64             // bytes; 0 = no limit
	NanoCPUs int64             // CPU limit in units of 1e-9 CPUs (e.g. 2e9 = 2 cores)
	Network  string            // "none", "bridge", etc.
	Tmpfs    map[string]string // e.g. {"/testbed": "exec,size=4g"}
	Runtime  string            // "crun" or "" for default
	Platform string            // "linux/amd64" or "linux/arm64"
}

// ExecResult holds the result of executing a command in a container.
type ExecResult struct {
	Stdout   string
	Stderr   string
	ExitCode int
}

// NewClient creates a new Docker SDK client configured from the environment.
// It auto-detects the Docker socket by checking DOCKER_HOST, then well-known
// socket paths (Colima, Rancher, Podman, default) to avoid stale symlinks.
func NewClient() (*Client, error) {
	opts := []client.Opt{
		client.FromEnv,
		client.WithAPIVersionNegotiation(),
	}

	// If DOCKER_HOST is not set, probe well-known socket paths so we don't
	// rely on /var/run/docker.sock which may be a stale Desktop symlink.
	var host string
	if os.Getenv("DOCKER_HOST") == "" {
		if sock := detectSocket(); sock != "" {
			host = "unix://" + sock
			opts = append(opts, client.WithHost(host))
		}
	} else {
		host = os.Getenv("DOCKER_HOST")
	}

	api, err := client.NewClientWithOpts(opts...)
	if err != nil {
		return nil, fmt.Errorf("docker client: %w", err)
	}

	return &Client{api: api, host: host}, nil
}

// detectSocket returns the first reachable Docker socket from well-known paths.
func detectSocket() string {
	home, _ := os.UserHomeDir()
	candidates := []string{
		home + "/.colima/default/docker.sock",
		home + "/.colima/docker.sock",
		home + "/.rd/docker.sock",         // Rancher Desktop
		home + "/.local/share/containers/podman/machine/podman.sock",
		"/var/run/docker.sock",
	}
	for _, sock := range candidates {
		if fi, err := os.Stat(sock); err == nil && fi.Mode().Type() == os.ModeSocket {
			return sock
		}
	}
	return ""
}

// Close releases the Docker client resources.
func (c *Client) Close() error {
	return c.api.Close()
}

// ServerArch returns the Docker daemon's architecture (e.g., "x86_64", "aarch64").
func (c *Client) ServerArch(ctx context.Context) string {
	info, err := c.api.Info(ctx)
	if err != nil {
		return "x86_64" // fallback
	}
	return info.Architecture
}

// ImageExists checks if an image exists locally.
func (c *Client) ImageExists(ctx context.Context, ref string) (bool, error) {
	opts := image.ListOptions{
		Filters: filters.NewArgs(filters.Arg("reference", ref)),
	}
	images, err := c.api.ImageList(ctx, opts)
	if err != nil {
		return false, fmt.Errorf("image list: %w", err)
	}
	return len(images) > 0, nil
}

// ImagePull pulls an image from a registry, discarding progress output.
func (c *Client) ImagePull(ctx context.Context, ref string) error {
	reader, err := c.api.ImagePull(ctx, ref, image.PullOptions{})
	if err != nil {
		return fmt.Errorf("image pull %s: %w", ref, err)
	}
	defer reader.Close()
	// Drain output to complete the pull
	_, _ = io.Copy(io.Discard, reader)
	return nil
}

// ImageBuild builds an image from a tar build context using docker buildx.
// This supports cross-platform builds (e.g. linux/amd64 on ARM hosts via QEMU).
func (c *Client) ImageBuild(ctx context.Context, buildCtx io.Reader, tag, dockerfile, platform string) error {
	args := []string{"buildx", "build",
		"--load",
		"-t", tag,
		"-f", dockerfile,
	}
	if platform != "" {
		args = append(args, "--platform", platform)
	}
	args = append(args, "-") // read tar context from stdin

	cmd := exec.CommandContext(ctx, "docker", args...)
	cmd.Stdin = buildCtx
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// Pass resolved Docker host so buildx talks to the right daemon.
	if c.host != "" {
		cmd.Env = append(os.Environ(), "DOCKER_HOST="+c.host)
	}

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("image build %s: %w", tag, err)
	}
	return nil
}

// ImagePush pushes a local image to a registry using docker push.
func (c *Client) ImagePush(ctx context.Context, ref string) error {
	args := []string{"push", ref}
	cmd := exec.CommandContext(ctx, "docker", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if c.host != "" {
		cmd.Env = append(os.Environ(), "DOCKER_HOST="+c.host)
	}
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("image push %s: %w", ref, err)
	}
	return nil
}

// ImageTag tags a local image with a new reference.
func (c *Client) ImageTag(ctx context.Context, source, target string) error {
	return c.api.ImageTag(ctx, source, target)
}

// ImageRemove removes a local image by reference.
func (c *Client) ImageRemove(ctx context.Context, ref string) error {
	_, err := c.api.ImageRemove(ctx, ref, image.RemoveOptions{PruneChildren: true})
	if err != nil {
		return fmt.Errorf("image remove %s: %w", ref, err)
	}
	return nil
}

// ContainerCreate creates a new container and returns its ID.
func (c *Client) ContainerCreate(ctx context.Context, opts ContainerOpts) (string, error) {
	cmd := opts.Cmd
	if len(cmd) == 0 {
		cmd = []string{"sleep", "infinity"}
	}
	cfg := &container.Config{
		Image: opts.Image,
		Cmd:   cmd,
	}

	hostCfg := &container.HostConfig{}

	if opts.MemLimit > 0 {
		hostCfg.Resources.Memory = opts.MemLimit
	}

	if opts.NanoCPUs > 0 {
		hostCfg.Resources.NanoCPUs = opts.NanoCPUs
	}

	if opts.Network != "" {
		hostCfg.NetworkMode = container.NetworkMode(opts.Network)
	}

	if len(opts.Tmpfs) > 0 {
		hostCfg.Tmpfs = opts.Tmpfs
	}

	if opts.Runtime != "" {
		hostCfg.Runtime = opts.Runtime
	}

	var platform *ocispec.Platform
	if opts.Platform != "" {
		parts := strings.SplitN(opts.Platform, "/", 2)
		if len(parts) == 2 {
			platform = &ocispec.Platform{
				OS:           parts[0],
				Architecture: parts[1],
			}
		}
	}

	resp, err := c.api.ContainerCreate(ctx, cfg, hostCfg, nil, platform, opts.Name)
	if err != nil {
		return "", fmt.Errorf("container create: %w", err)
	}
	return resp.ID, nil
}

// ContainerStart starts a stopped container.
func (c *Client) ContainerStart(ctx context.Context, id string) error {
	if err := c.api.ContainerStart(ctx, id, container.StartOptions{}); err != nil {
		return fmt.Errorf("container start %s: %w", id, err)
	}
	return nil
}

// ContainerExec runs a command inside a running container and returns the result.
func (c *Client) ContainerExec(ctx context.Context, id string, cmd []string, timeout time.Duration) (ExecResult, error) {
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	execCfg := container.ExecOptions{
		Cmd:          cmd,
		AttachStdout: true,
		AttachStderr: true,
	}

	execResp, err := c.api.ContainerExecCreate(ctx, id, execCfg)
	if err != nil {
		return ExecResult{}, fmt.Errorf("exec create: %w", err)
	}

	attachResp, err := c.api.ContainerExecAttach(ctx, execResp.ID, container.ExecAttachOptions{})
	if err != nil {
		return ExecResult{}, fmt.Errorf("exec attach: %w", err)
	}
	defer attachResp.Close()

	var stdoutBuf, stderrBuf bytes.Buffer
	_, err = stdcopy.StdCopy(&stdoutBuf, &stderrBuf, attachResp.Reader)
	if err != nil {
		return ExecResult{}, fmt.Errorf("exec read: %w", err)
	}

	inspect, err := c.api.ContainerExecInspect(ctx, execResp.ID)
	if err != nil {
		return ExecResult{}, fmt.Errorf("exec inspect: %w", err)
	}

	return ExecResult{
		Stdout:   stdoutBuf.String(),
		Stderr:   stderrBuf.String(),
		ExitCode: inspect.ExitCode,
	}, nil
}

// ContainerExecStream runs a command inside a container, streaming output to the provided writers.
func (c *Client) ContainerExecStream(ctx context.Context, id string, cmd []string, timeout time.Duration, stdout, stderr io.Writer) (int, error) {
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	execCfg := container.ExecOptions{
		Cmd:          cmd,
		AttachStdout: true,
		AttachStderr: true,
	}

	execResp, err := c.api.ContainerExecCreate(ctx, id, execCfg)
	if err != nil {
		return -1, fmt.Errorf("exec create: %w", err)
	}

	attachResp, err := c.api.ContainerExecAttach(ctx, execResp.ID, container.ExecAttachOptions{})
	if err != nil {
		return -1, fmt.Errorf("exec attach: %w", err)
	}
	defer attachResp.Close()

	_, err = stdcopy.StdCopy(stdout, stderr, attachResp.Reader)
	if err != nil {
		return -1, fmt.Errorf("exec stream: %w", err)
	}

	inspect, err := c.api.ContainerExecInspect(ctx, execResp.ID)
	if err != nil {
		return -1, fmt.Errorf("exec inspect: %w", err)
	}

	return inspect.ExitCode, nil
}

// CopyToContainer copies content bytes to a file at destPath inside the container.
// Builds the tar archive in-memory (no temp files).
func (c *Client) CopyToContainer(ctx context.Context, id string, content []byte, destPath string) error {
	dir := filepath.Dir(destPath)
	base := filepath.Base(destPath)

	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)

	hdr := &tar.Header{
		Name: base,
		Mode: 0644,
		Size: int64(len(content)),
	}
	if err := tw.WriteHeader(hdr); err != nil {
		return fmt.Errorf("tar header: %w", err)
	}
	if _, err := tw.Write(content); err != nil {
		return fmt.Errorf("tar write: %w", err)
	}
	if err := tw.Close(); err != nil {
		return fmt.Errorf("tar close: %w", err)
	}

	return c.api.CopyToContainer(ctx, id, dir, &buf, container.CopyToContainerOptions{})
}

// ContainerRemove force-removes a container.
func (c *Client) ContainerRemove(ctx context.Context, id string) error {
	return c.api.ContainerRemove(ctx, id, container.RemoveOptions{Force: true})
}

// ParseMemLimit converts a human-readable memory limit string (e.g., "4g") to bytes.
func ParseMemLimit(s string) (int64, error) {
	if s == "" {
		return 0, nil
	}
	return units.RAMInBytes(s)
}
