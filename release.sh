#!/bin/bash
# Automate Git Flow release with tag auto-increment and tests
# Usage: ./release.sh [major|minor|patch]

set -e
set -o pipefail

# --- CONFIG ---
INCREMENT_PART=${1:-"patch"} # "patch", "minor", or "major"
TEST_CMD="pytest tests/ -q"  # can also be: poetry run pytest tests/

echo "🚀 Starting weekly release..."

# --- Ensure clean repo ---
if [ -n "$(git status --porcelain)" ]; then
  echo "⚠️ Working directory not clean. Commit or stash changes first."
  exit 1
fi

git fetch origin --tags
git checkout develop
git pull origin develop

# --- Determine last tag ---
LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
echo "🔖 Last tag: ${LAST_TAG}"

# --- Parse and increment ---
IFS='.' read -r MAJOR MINOR PATCH <<<"${LAST_TAG#v}"

case $INCREMENT_PART in
major)
  MAJOR=$((MAJOR + 1))
  MINOR=0
  PATCH=0
  ;;
minor)
  MINOR=$((MINOR + 1))
  PATCH=0
  ;;
patch)
  PATCH=$((PATCH + 1))
  ;;
*)
  echo "❌ Invalid INCREMENT_PART: $INCREMENT_PART"
  exit 1
  ;;
esac

NEW_TAG="v${MAJOR}.${MINOR}.${PATCH}"
echo "🏷️ New tag: ${NEW_TAG}"

# --- Start release branch ---
git flow release start "${NEW_TAG}"

# Optionally bump version in pyproject.toml:
# sed -i "s/^version = .*/version = \"${NEW_TAG}\"/" pyproject.toml
# git commit -am "Bump version to ${NEW_TAG}"

# --- Run tests ---
echo "🧪 Running tests..."
if $TEST_CMD; then
  echo "✅ Tests passed!"
else
  echo "❌ Tests failed! Aborting release."
  git flow release delete "${NEW_TAG}" -f
  exit 1
fi

# --- Finish release ---
git flow release finish -m "Weekly release ${NEW_TAG}" "${NEW_TAG}"

# --- Push changes ---
git push origin develop master --follow-tags

echo "🎉 Release ${NEW_TAG} complete and pushed successfully!"
