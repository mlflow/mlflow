#!/usr/bin/env bash
# Type-check each integration against the *lowest published version* of every
# @mlflow/* workspace dependency its package.json allows.
#
# Why this exists: in the workspace, node_modules/@mlflow/core is a symlink to
# ../../core (always the latest source), so `tsc`, eslint, and jest all resolve
# @mlflow/core to workspace-latest. That means an integration can import a symbol
# that only exists in the unreleased core and still pass every CI gate -- while a
# user running `npm install @mlflow/opencode @mlflow/core` gets the published
# floor, where the symbol is missing, and crashes at runtime (see mlflow#24167).
#
# This gate reproduces the user's install: for each @mlflow/* dependency, it
# resolves the lowest version the declared range allows, installs that published
# version into the integration's own node_modules (shadowing the symlink), and
# runs `tsc --noEmit`. A newer-than-floor symbol becomes a hard compile error.
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
  if [[ -e "$shadow" ]]; then
    backup="$(mktemp -d)/node_modules"
    mv "$shadow" "$backup"
  fi
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
    # Install the published floor into a temp prefix, then copy it into the
    # integration's node_modules so Node/tsc resolution finds it before the
    # hoisted workspace symlink.
    tmp="$(mktemp -d)"
    # Silence normal progress on stdout but keep stderr, so a failed install
    # (unpublished version, network error) surfaces its reason in CI logs.
    if ! npm install --prefix "$tmp" "$dep@$floor" \
      --no-save --no-package-lock --no-audit --no-fund >/dev/null; then
      echo "        ERROR: failed to install $dep@$floor (see npm output above)" >&2
      rm -rf "$tmp" "$shadow"
      [[ -n "$backup" ]] && mv "$backup" "$shadow" && rmdir "$(dirname "$backup")"
      exit 1
    fi
    rm -rf "$shadow/$dep"
    cp -R "$tmp/node_modules/$dep" "$shadow/$dep"
    rm -rf "$tmp"
  done <<< "$mlflow_deps"

  # At-floor errors minus baseline errors == breakage caused purely by the floor.
  floor_errors="$(errors_here || true)"
  violations="$(comm -13 <(printf '%s\n' "$baseline") <(printf '%s\n' "$floor_errors"))"
  rm -rf "$shadow"
  if [[ -n "$backup" ]]; then
    mv "$backup" "$shadow"
    rmdir "$(dirname "$backup")"
  fi

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
