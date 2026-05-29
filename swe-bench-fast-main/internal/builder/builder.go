package builder

import (
	"archive/tar"
	"bytes"
	"context"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/greynewell/swe-bench-fast/internal/docker"
	"github.com/greynewell/swe-bench-fast/internal/spec"
)

// BuildConfig holds configuration for a build run.
type BuildConfig struct {
	Client       *docker.Client
	Workers      int
	ForceRebuild bool
	Arch         string
	// PostBuild is called after each instance image is built successfully.
	// Use this for push-and-delete workflows. Called with the spec of the built image.
	PostBuild func(ctx context.Context, s spec.ImageSpec) error
}

// BuildResult holds the result of building a single image.
type BuildResult struct {
	Tag      string
	Cached   bool
	Duration time.Duration
	Error    error
}

// BuildAll builds all images in the 3-layer pipeline: base → env → instance.
// It deduplicates shared layers and builds in parallel.
func BuildAll(ctx context.Context, cfg BuildConfig, specs []spec.ImageSpec) ([]BuildResult, error) {
	if cfg.Workers <= 0 {
		cfg.Workers = 4
	}

	// Phase 1: Build base images (1-2 unique, sequential)
	baseTags := dedupStrings(specs, func(s spec.ImageSpec) string { return s.BaseTag })
	fmt.Printf("Building %d base image(s)...\n", len(baseTags))
	for _, tag := range baseTags {
		if err := buildBase(ctx, cfg, tag); err != nil {
			return nil, fmt.Errorf("build base %s: %w", tag, err)
		}
	}

	// Phase 2: Build env images (~60 unique, parallel)
	envTags := dedupStrings(specs, func(s spec.ImageSpec) string { return s.EnvTag })
	envSpecMap := make(map[string]spec.ImageSpec)
	for _, s := range specs {
		if _, ok := envSpecMap[s.EnvTag]; !ok {
			envSpecMap[s.EnvTag] = s
		}
	}
	fmt.Printf("Building %d env image(s) with %d workers...\n", len(envTags), cfg.Workers)
	envResults := buildParallel(ctx, cfg, envTags, func(ctx context.Context, tag string) error {
		s := envSpecMap[tag]
		return buildEnv(ctx, cfg, s)
	})
	failedEnvs := make(map[string]bool)
	for _, r := range envResults {
		if r.Error != nil {
			fmt.Printf("  ENV FAILED %s: %v\n", r.Tag, r.Error)
			failedEnvs[r.Tag] = true
		}
	}

	// Phase 3: Build instance images (all, parallel), skipping those with failed envs
	var buildableSpecs []spec.ImageSpec
	var skippedEnv int
	for _, s := range specs {
		if failedEnvs[s.EnvTag] {
			skippedEnv++
			continue
		}
		buildableSpecs = append(buildableSpecs, s)
	}
	if skippedEnv > 0 {
		fmt.Printf("Skipping %d instance(s) due to env build failures\n", skippedEnv)
	}
	fmt.Printf("Building %d instance image(s) with %d workers...\n", len(buildableSpecs), cfg.Workers)
	instanceTags := make([]string, len(buildableSpecs))
	instanceSpecMap := make(map[string]spec.ImageSpec)
	for i, s := range buildableSpecs {
		instanceTags[i] = s.InstanceTag
		instanceSpecMap[s.InstanceTag] = s
	}
	results := buildParallel(ctx, cfg, instanceTags, func(ctx context.Context, tag string) error {
		s := instanceSpecMap[tag]
		if err := buildInstance(ctx, cfg, s); err != nil {
			return err
		}
		if cfg.PostBuild != nil {
			return cfg.PostBuild(ctx, s)
		}
		return nil
	})

	return results, nil
}

// archToPlatform converts a server architecture string to a Docker platform string.
func archToPlatform(arch string) string {
	if arch == "aarch64" || arch == "arm64" {
		return "linux/arm64"
	}
	return "linux/amd64"
}

func buildBase(ctx context.Context, cfg BuildConfig, tag string) error {
	if !cfg.ForceRebuild {
		exists, err := cfg.Client.ImageExists(ctx, tag)
		if err != nil {
			return err
		}
		if exists {
			fmt.Printf("  [cached] %s\n", tag)
			return nil
		}
	}

	dockerfile := spec.GenerateBaseDockerfile(cfg.Arch)
	buildCtx := makeBuildContext(map[string][]byte{
		"Dockerfile": []byte(dockerfile),
	})

	fmt.Printf("  [build] %s\n", tag)
	return cfg.Client.ImageBuild(ctx, buildCtx, tag, "Dockerfile", archToPlatform(cfg.Arch))
}

func buildEnv(ctx context.Context, cfg BuildConfig, s spec.ImageSpec) error {
	if !cfg.ForceRebuild {
		exists, err := cfg.Client.ImageExists(ctx, s.EnvTag)
		if err != nil {
			return err
		}
		if exists {
			return nil
		}
	}

	dockerfile := spec.GenerateEnvDockerfile(s.BaseTag)
	setupEnv := spec.GenerateSetupEnvScript(s)
	buildCtx := makeBuildContext(map[string][]byte{
		"Dockerfile":   []byte(dockerfile),
		"setup_env.sh": []byte(setupEnv),
	})

	return cfg.Client.ImageBuild(ctx, buildCtx, s.EnvTag, "Dockerfile", archToPlatform(cfg.Arch))
}

func buildInstance(ctx context.Context, cfg BuildConfig, s spec.ImageSpec) error {
	if !cfg.ForceRebuild {
		exists, err := cfg.Client.ImageExists(ctx, s.InstanceTag)
		if err != nil {
			return err
		}
		if exists {
			return nil
		}
	}

	dockerfile := spec.GenerateInstanceDockerfile(s.EnvTag)
	setupRepo := spec.GenerateSetupRepoScript(s)
	buildCtx := makeBuildContext(map[string][]byte{
		"Dockerfile":    []byte(dockerfile),
		"setup_repo.sh": []byte(setupRepo),
	})

	return cfg.Client.ImageBuild(ctx, buildCtx, s.InstanceTag, "Dockerfile", archToPlatform(cfg.Arch))
}

// makeBuildContext creates an in-memory tar archive from a map of filename → content.
func makeBuildContext(files map[string][]byte) io.Reader {
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)

	for name, content := range files {
		hdr := &tar.Header{
			Name: name,
			Mode: 0644,
			Size: int64(len(content)),
		}
		_ = tw.WriteHeader(hdr)
		_, _ = tw.Write(content)
	}
	_ = tw.Close()

	return &buf
}

// buildParallel runs build functions in parallel with a worker limit.
func buildParallel(ctx context.Context, cfg BuildConfig, tags []string, fn func(ctx context.Context, tag string) error) []BuildResult {
	results := make([]BuildResult, len(tags))
	sem := make(chan struct{}, cfg.Workers)
	var wg sync.WaitGroup

	for i, tag := range tags {
		wg.Add(1)
		go func(i int, tag string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			start := time.Now()
			err := fn(ctx, tag)
			results[i] = BuildResult{
				Tag:      tag,
				Cached:   err == nil && time.Since(start) < time.Second,
				Duration: time.Since(start),
				Error:    err,
			}
		}(i, tag)
	}

	wg.Wait()
	return results
}

// dedupStrings extracts unique strings from specs using a key function.
func dedupStrings(specs []spec.ImageSpec, key func(spec.ImageSpec) string) []string {
	seen := make(map[string]bool)
	var result []string
	for _, s := range specs {
		k := key(s)
		if !seen[k] {
			seen[k] = true
			result = append(result, k)
		}
	}
	return result
}
