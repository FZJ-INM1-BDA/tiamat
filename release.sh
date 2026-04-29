#!/bin/bash
# Automate Git Flow release with tag auto-increment and tests
# Usage: ./release.sh [major|minor|patch]

set -e
set -o pipefail

INCREMENT_PART="patch"
TEST_CMD="pytest tests/ -q"

echo "🚀 Starting weekly release..."

# --- Ensure clean repo ---
if [ -n "$(git status --porcelain)" ]; then
  echo "⚠️ Working directory not clean. Commit or stash changes first."
  exit 1
fi

git fetch origin --tags
git checkout develop
git pull origin develop

# --- Determine new tag ---
LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
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
patch) PATCH=$((PATCH + 1)) ;;
esac

NEW_TAG="v${MAJOR}.${MINOR}.${PATCH}"
RELEASE_BRANCH="release/${NEW_TAG}"

echo "🔖 Creating release branch ${RELEASE_BRANCH}"

git flow release start "${NEW_TAG}"

# --- Run tests ---
echo "🧪 Running tests..."
if $TEST_CMD; then
  echo "✅ Tests passed!"
else
  echo "❌ Tests failed! Aborting release."
  git flow release delete "${NEW_TAG}" -f
  exit 1
fi

# --- Write metadata to codemeta.json ---
export ymd=$(date '+%Y-%m-%d')
jq --indent 4 \
    --arg vsn "$NEW_TAG" \
    --arg ymd "$ymd" \
    '
        .version=$vsn,
        .dateModified=$ymd,
        .datePublished=$ymd
    ' codemeta.json  > _codemeta.json \
&& mv _codemeta.json codemeta.json

git add codemeta.json
git commit -m "Update codemta.json metadata"

# --- Merge into develop locally ---
git checkout develop
git merge --no-ff "${RELEASE_BRANCH}" -m "Merge ${RELEASE_BRANCH} into develop"

# --- Tag the release ---
git tag -a "${NEW_TAG}" -m "Release ${NEW_TAG}"

# --- Push to origin (but NOT to master) ---
git push origin develop "${RELEASE_BRANCH}" --follow-tags

echo "📤 Release branch ${RELEASE_BRANCH} and tag ${NEW_TAG} pushed."

echo "📋 Next step: open a PR from ${RELEASE_BRANCH} → master in your repo UI."
echo "🎉 Done!"
