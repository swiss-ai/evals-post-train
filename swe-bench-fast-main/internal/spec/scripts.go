package spec

import (
	"fmt"
	"strings"
)

// GenerateSetupEnvScript generates the script that creates the conda environment
// and installs dependencies for a given spec.
// Matches upstream make_env_script_list_py.
func GenerateSetupEnvScript(s ImageSpec) string {
	var b strings.Builder
	b.WriteString("#!/bin/bash\nset -euxo pipefail\n\n")
	b.WriteString("source /opt/miniconda3/bin/activate\n")

	pkg := s.RepoSpec.Packages
	switch {
	case pkg == "requirements.txt":
		// Create environment, then install from requirements
		b.WriteString(fmt.Sprintf("conda create -n testbed python=%s -y\n", s.RepoSpec.Python))

		// Upstream fetches requirements.txt from GitHub at the base commit.
		// We install from repo files in the repo setup script (after clone).
		// Here we just create the environment.

	case pkg == "environment.yml":
		if s.RepoSpec.NoUseEnv {
			// conda create based installation
			b.WriteString(fmt.Sprintf("conda create -c conda-forge -n testbed python=%s -y\n", s.RepoSpec.Python))
			// environment.yml will be applied in repo setup script
		} else {
			// conda env create based installation — handled in repo setup script
			// Just create a placeholder here; the env is fully set up after clone.
			b.WriteString(fmt.Sprintf("conda create -n testbed python=%s -y\n", s.RepoSpec.Python))
		}

	default:
		// Inline packages: conda create with packages
		if pkg != "" {
			b.WriteString(fmt.Sprintf("conda create -n testbed python=%s %s -y\n", s.RepoSpec.Python, pkg))
		} else {
			b.WriteString(fmt.Sprintf("conda create -n testbed python=%s -y\n", s.RepoSpec.Python))
		}
	}

	b.WriteString("conda activate testbed\n\n")

	if len(s.RepoSpec.PipPackages) > 0 {
		var quoted []string
		for _, pkg := range s.RepoSpec.PipPackages {
			quoted = append(quoted, fmt.Sprintf("'%s'", pkg))
		}
		b.WriteString(fmt.Sprintf("python -m pip install %s\n", strings.Join(quoted, " ")))
	}

	return b.String()
}

// GenerateSetupRepoScript generates the script that clones the repo,
// checks out the base commit, and runs the install command.
// Matches upstream make_repo_script_list_py.
func GenerateSetupRepoScript(s ImageSpec) string {
	var b strings.Builder
	b.WriteString("#!/bin/bash\nset -euxo pipefail\n\n")

	repoURL := fmt.Sprintf("https://github.com/%s.git", s.Repo)
	repoDir := "/testbed"

	b.WriteString("cd /\n")
	b.WriteString(fmt.Sprintf("rm -rf %s\n", repoDir))
	b.WriteString(fmt.Sprintf("git clone -o origin --single-branch %s %s\n", repoURL, repoDir))
	b.WriteString(fmt.Sprintf("chmod -R 777 %s\n", repoDir))
	b.WriteString(fmt.Sprintf("cd %s\n", repoDir))
	b.WriteString(fmt.Sprintf("git reset --hard %s\n", s.BaseCommit))

	// Remove the remote and future tags so the agent won't see newer commits
	b.WriteString("git remote remove origin\n")
	b.WriteString(fmt.Sprintf("TARGET_TIMESTAMP=$(git show -s --format=%%ci %s)\n", s.BaseCommit))
	b.WriteString(`git tag -l | while read tag; do TAG_COMMIT=$(git rev-list -n 1 "$tag"); TAG_TIME=$(git show -s --format=%ci "$TAG_COMMIT"); if [[ "$TAG_TIME" > "$TARGET_TIMESTAMP" ]]; then git tag -d "$tag"; fi; done` + "\n")
	b.WriteString("git reflog expire --expire=now --all\n")
	b.WriteString("git gc --prune=now --aggressive\n")

	// Verify future logs aren't available
	b.WriteString(`AFTER_TIMESTAMP=$(date -d "$TARGET_TIMESTAMP + 1 second" '+%Y-%m-%d %H:%M:%S')` + "\n")
	b.WriteString(`COMMIT_COUNT=$(git log --oneline --all --since="$AFTER_TIMESTAMP" | wc -l)` + "\n")
	b.WriteString(`[ "$COMMIT_COUNT" -eq 0 ] || exit 1` + "\n\n")

	// Activate conda environment
	b.WriteString("source /opt/miniconda3/bin/activate\n")
	b.WriteString("conda activate testbed\n")
	b.WriteString("echo \"Current environment: $CONDA_DEFAULT_ENV\"\n\n")

	// Install packages from repo files (after checkout, before pre_install)
	if s.RepoSpec.Packages == "requirements.txt" {
		if paths, ok := repoReqsPaths[s.Repo]; ok {
			for _, p := range paths {
				b.WriteString(fmt.Sprintf("if [ -f %s ]; then python -m pip install -r %s; fi\n", p, p))
			}
		}
	} else if s.RepoSpec.Packages == "environment.yml" {
		if paths, ok := repoEnvYMLPaths[s.Repo]; ok {
			for _, p := range paths {
				if s.RepoSpec.NoUseEnv {
					b.WriteString(fmt.Sprintf("if [ -f %s ]; then conda env update -f %s; fi\n", p, p))
				} else {
					b.WriteString(fmt.Sprintf("if [ -f %s ]; then conda env update --file %s; fi\n", p, p))
				}
			}
		}
	}

	// Run pre-install commands
	for _, cmd := range s.RepoSpec.PreInstall {
		b.WriteString(cmd + "\n")
	}

	// Run install command
	if s.RepoSpec.Install != "" {
		b.WriteString(s.RepoSpec.Install + "\n")
	}

	// Create a clean git commit — matches upstream
	b.WriteString("\ngit config --global user.email setup@swebench.config\n")
	b.WriteString("git config --global user.name SWE-bench\n")
	b.WriteString("git commit --allow-empty -am SWE-bench\n")

	return b.String()
}
