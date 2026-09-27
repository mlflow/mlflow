#!/usr/bin/env bash
# Type-check each integration against the lowest version of every @mlflow/*
# workspace dependency its package.json allows.
#
# Why this exists: in the workspace, node_modules/@mlflow/core is a symlink to
# ../../core (always the latest source), so `tsc`, eslint, and jest all resolve
# @mlflow/core to workspace-latest. That means an integration can import a symbol
# that only exists in the unreleased core and still pass every CI gate -- while a
# user running `npm install @mlflow/opencode @mlflow/core` gets the published
# floor, where the symbol is missing, and crashes at runtime (see mlflow#24167).
#
# This gate reproduces the user's install: for each @mlflow/* dependency, it
# resolves the lowest version the declared range allows and installs it into the
# integration's own node_modules (shadowing the symlink). Published floors come
# from npm. During a coordinated release, an unpublished floor may instead come
# from a packed local workspace whose name and version match exactly. Published
# floors retain the compatibility comparison described below. For an unpublished
# workspace floor, the baseline and package come from the same source, so the
# check only verifies that the package builds, installs, and type-checks; it
# cannot detect API-floor violations until that version is published.
#
# To stay immune to unrelated noise (a third-party module missing from a local
# sandbox, a pre-existing type error), it type-checks TWICE -- once against
# workspace-latest (the baseline) and once against the floor -- and treats only
# errors that appear *exclusively* against the floor as floor violations. The
# source is identical between runs, so error lines match exactly and a plain
# set-difference isolates the version-specific breakage.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
TSC="$ROOT/node_modules/.bin/tsc"
ALLOWLIST="$ROOT/scripts/known-floor-violations.txt"
LOCAL_PACKAGE_CACHE="$(mktemp -d)"
shadow=""
backup=""
tmp=""
shadow_active=false

restore_shadow() {
  local backup_dir

  if [[ -n "$tmp" ]]; then
    rm -rf "$tmp"
    tmp=""
  fi
  if [[ "$shadow_active" != true ]]; then
    return 0
  fi

  rm -rf "$shadow"
  if [[ -n "$backup" && -e "$backup" ]]; then
    backup_dir="$(dirname "$backup")"
    mv "$backup" "$shadow"
    rmdir "$backup_dir"
  fi
  shadow=""
  backup=""
  shadow_active=false
}

cleanup() {
  restore_shadow
  rm -rf "$LOCAL_PACKAGE_CACHE"
}

trap cleanup EXIT

# Integrations grandfathered in: they violate the floor today (same class as
# mlflow#24167) but must not fail the gate yet. New violations still fail.
is_allowlisted() {
  [[ -f "$ALLOWLIST" ]] || return 1
  grep -vE '^[[:space:]]*(#|$)' "$ALLOWLIST" | grep -qxF "$1"
}

# Append markdown to the GitHub Actions run summary when available (a no-op
# locally). Renders a result table on the run's page needing no token.
summary() {
  [[ -n "${GITHUB_STEP_SUMMARY:-}" ]] && printf '%s\n' "$1" >> "$GITHUB_STEP_SUMMARY"
  return 0
}

published_version_exists() {
  local dep="$1"
  local version="$2"
  local versions

  if ! versions="$(npm view "$dep" versions --json)"; then
    echo "        ERROR: failed to query published versions for $dep" >&2
    return 2
  fi

  node -e '
    const fs = require("fs");
    const value = JSON.parse(fs.readFileSync(0, "utf8"));
    const versions = Array.isArray(value) ? value : [value];
    process.exit(versions.includes(process.argv[1]) ? 0 : 1);
  ' "$version" <<< "$versions"
}

workspace_package_path() {
  node -e '
    const lock = require("./package-lock.json");
    const name = process.argv[1];
    const match = Object.entries(lock.packages).find(
      ([path, pkg]) => path && !path.startsWith("node_modules/") && pkg.name === name,
    );
    if (match) process.stdout.write(match[0]);
  ' "$1"
}

pack_workspace_package() {
  local dep="$1"
  local version="$2"
  local workspace_path="$3"
  local cache_key="${dep//@/_}"
  local pack_json
  local packed_filename
  local tarball

  cache_key="${cache_key//\//_}"
  tarball="$LOCAL_PACKAGE_CACHE/$cache_key-$version.tgz"
  if [[ ! -f "$tarball" ]]; then
    echo "        BUILD $dep@$version from workspace $workspace_path" >&2
    if ! npm run -C "$workspace_path" build >&2; then
      echo "        ERROR: failed to build workspace package $dep@$version" >&2
      return 1
    fi
    if ! pack_json="$(npm pack "./$workspace_path" --pack-destination "$LOCAL_PACKAGE_CACHE" --json)"; then
      echo "        ERROR: failed to pack workspace package $dep@$version" >&2
      return 1
    fi
    if ! packed_filename="$(node -e '
      const fs = require("fs");
      const result = JSON.parse(fs.readFileSync(0, "utf8"));
      const packed = Array.isArray(result) ? result : Object.values(result);
      if (typeof packed[0]?.filename !== "string") process.exit(1);
      process.stdout.write(packed[0].filename);
    ' <<< "$pack_json")"; then
      echo "        ERROR: failed to parse npm pack output for $dep@$version" >&2
      return 1
    fi
    if ! mv "$LOCAL_PACKAGE_CACHE/$packed_filename" "$tarball"; then
      echo "        ERROR: failed to cache packed workspace package $dep@$version" >&2
      return 1
    fi
  fi

  printf '%s' "$tarball"
}

new_failures=()      # not allowlisted -> fail the build
known_failures=()    # allowlisted -> reported only
stale_allowlist=()   # allowlisted but now passing -> fail, asking to remove
summary "### Minimum-version floor check"
summary "| Integration | Result |"
summary "| --- | --- |"

for pkg_json in integrations/*/package.json; do
  dir="$(dirname "$pkg_json")"
  slug="$(basename "$dir")"
  name="$(node -p "require('./$pkg_json').name")"

  mlflow_deps="$(node -p "
    const d = require('./$pkg_json').dependencies || {};
    Object.keys(d).filter((k) => k.startsWith('@mlflow/')).join('\n')
  ")"
  if [[ -z "$mlflow_deps" ]]; then
    echo "SKIP  $name (no @mlflow/* dependency)"
    continue
  fi

  echo "CHECK $name"

  # tsc error lines with the source dir stripped, so the two runs are comparable.
  errors_here() { (cd "$dir" && "$TSC" --noEmit 2>&1) | grep -E 'error TS[0-9]+' | sort -u; }

  # Baseline: workspace-latest (the symlinked source), no shadowing.
  baseline="$(errors_here || true)"

  # Shadow the floor into the integration's own node_modules. Preserve any
  # pre-existing one (common when iterating locally) and restore it at the end
  # so the check never leaves the working tree modified.
  shadow="$dir/node_modules"
  backup=""
  tmp=""
  shadow_active=false
  if [[ -e "$shadow" ]]; then
    backup_dir="$(mktemp -d)"
    if ! mv "$shadow" "$backup_dir/node_modules"; then
      rmdir "$backup_dir"
      exit 1
    fi
    backup="$backup_dir/node_modules"
  fi
  shadow_active=true
  mkdir -p "$shadow/@mlflow"
  while IFS= read -r dep; do
    [[ -z "$dep" ]] && continue
    range="$(node -p "require('./$pkg_json').dependencies['$dep']")"
    # Lowest version the range admits. Handles the forms actually used here:
    # "^x.y.z", "~x.y.z", ">=x.y.z", "x.y.z", and "A || B" (take the min of each
    # clause's floor). Deliberately dependency-free -- semver is only a
    # transitive package here and adding it would churn the lockfile.
    floor="$(node -e "
      const range = process.argv[1];
      const floors = range.split('||').map((clause) => {
        const m = clause.match(/(\d+)\.(\d+)\.(\d+)(?:-[0-9A-Za-z.-]+)?/);
        if (!m) throw new Error('Unparseable range: ' + range);
        return m[0];
      });
      const cmp = (a, b) => {
        const pa = a.split('-')[0].split('.').map(Number);
        const pb = b.split('-')[0].split('.').map(Number);
        for (let i = 0; i < 3; i++) if (pa[i] !== pb[i]) return pa[i] - pb[i];
        const ra = a.includes('-'), rb = b.includes('-');
        return ra === rb ? 0 : ra ? -1 : 1; // a prerelease sorts below its release
      };
      console.log(floors.sort(cmp)[0]);
    " "$range")"
    echo "        $dep  range='$range'  floor=$floor"
    # Prefer the published floor. For a coordinated release, fall back to a
    # packed workspace only when npm does not contain the floor and the local
    # package has the exact same name and version. This keeps ordinary feature
    # PRs pinned to published artifacts while allowing release PRs to validate
    # the package that will be published.
    install_spec="$dep@$floor"
    if published_version_exists "$dep" "$floor"; then
      echo "        SOURCE npm ($install_spec)"
    else
      status=$?
      if [[ $status -eq 2 ]]; then
        exit 1
      fi

      workspace_path="$(workspace_package_path "$dep")"
      if [[ -z "$workspace_path" ]]; then
        echo "        ERROR: $dep@$floor is unpublished and has no local workspace" >&2
        exit 1
      fi
      workspace_version="$(node -p "require('./$workspace_path/package.json').version")"
      if [[ "$workspace_version" != "$floor" ]]; then
        echo "        ERROR: $dep@$floor is unpublished and workspace version is $workspace_version" >&2
        exit 1
      fi

      if ! install_spec="$(pack_workspace_package "$dep" "$floor" "$workspace_path")"; then
        echo "        ERROR: failed to prepare workspace package $dep@$floor" >&2
        exit 1
      fi
      echo "        SOURCE workspace package ($workspace_path)"
    fi

    # Install the floor into a temp prefix, then copy it into the integration's
    # node_modules so Node/tsc resolution finds it before the hoisted symlink.
    tmp="$(mktemp -d)"
    # Silence normal progress on stdout but keep stderr so failures surface in
    # CI logs.
    if ! npm install --prefix "$tmp" "$install_spec" \
      --no-save --no-package-lock --no-audit --no-fund >/dev/null; then
      echo "        ERROR: failed to install $install_spec (see npm output above)" >&2
      exit 1
    fi
    rm -rf "$shadow/$dep"
    cp -R "$tmp/node_modules/$dep" "$shadow/$dep"
    rm -rf "$tmp"
    tmp=""
  done <<< "$mlflow_deps"

  # At-floor errors minus baseline errors == breakage caused purely by the floor.
  floor_errors="$(errors_here || true)"
  violations="$(comm -13 <(printf '%s\n' "$baseline") <(printf '%s\n' "$floor_errors"))"
  restore_shadow

  if [[ -z "$violations" ]]; then
    if is_allowlisted "$slug"; then
      echo "        OK: $name now type-checks against floors -- remove it from known-floor-violations.txt"
      stale_allowlist+=("$slug")
      summary "| \`$name\` | ⚠️ passes but still allowlisted — remove it |"
    else
      echo "        OK: $name type-checks against dependency floors"
      summary "| \`$name\` | ✅ ok |"
    fi
  elif is_allowlisted "$slug"; then
    echo "        KNOWN FAIL (allowlisted): $name -- pre-existing floor violation, not blocking"
    printf '%s\n' "$violations" | sed 's/^/          /'
    known_failures+=("$name")
    summary "| \`$name\` | 🕒 known violation (allowlisted) |"
  else
    echo "        FAIL: $name imports a symbol newer than its declared @mlflow/* floor"
    printf '%s\n' "$violations" | sed 's/^/          /'
    new_failures+=("$name")
    summary "| \`$name\` | ❌ NEW violation |"
  fi
done

echo
[[ ${#known_failures[@]} -gt 0 ]] && echo "Known (allowlisted) floor violations: ${known_failures[*]}"

if [[ ${#stale_allowlist[@]} -gt 0 ]]; then
  echo
  echo "These integrations are allowlisted but now pass -- delete them from"
  echo "scripts/known-floor-violations.txt so regressions are caught again: ${stale_allowlist[*]}"
  exit 1
fi

if [[ ${#new_failures[@]} -gt 0 ]]; then
  echo
  echo "Minimum-version check FAILED for: ${new_failures[*]}"
  echo "Each imports an @mlflow/* symbol missing from its declared floor, so a user"
  echo "running 'npm install' against the published dependency would crash at runtime."
  echo
  echo "To fix, first find out whether the symbol was ever published. Check the"
  echo "published versions with 'npm view @mlflow/core versions', then 'npm pack"
  echo "@mlflow/core@latest' and grep the extracted dist/index.d.ts (named import)"
  echo "or the relevant dist/**/*.d.ts (a member access like SomeConst.NEW_KEY):"
  echo "  - Symbol exists in a version newer than your floor -> raise the @mlflow/*"
  echo "    floor in the package's package.json to that version."
  echo "  - Symbol exists in NO published version -> raising the floor cannot help;"
  echo "    inline the value in the integration (see integrations/openclaw/src/service.ts)."
  echo "If this is knowingly deferred, add the integration to scripts/known-floor-violations.txt."
  exit 1
fi
echo "Minimum-version check passed (no new floor violations)."
