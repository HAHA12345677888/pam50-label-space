#!/usr/bin/env bash
# Create (if missing) and push this repository to GitHub.
#
# The token is read from the environment and never printed, never written to
# a file, and never embedded in a command line that would show up in the
# process list. Create it once under Customize -> Credentials -> Add
# Credential; the token then arrives as GITHUB_TOKEN in every session.
#
# Usage:
#   ./publish_github.sh            # dry run: report what would happen
#   ./publish_github.sh --push     # create the repo if needed, then push

set -euo pipefail

OWNER="${GITHUB_OWNER:-HAHA12345677888}"
NAME="${GITHUB_REPO:-breast-cancer-multimodal-transformer}"
BRANCH="${GITHUB_BRANCH:-main}"
REMOTE_URL="https://github.com/${OWNER}/${NAME}.git"

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "GITHUB_TOKEN is not set. Add a GitHub credential under"
  echo "Customize -> Credentials -> Add Credential, then re-run."
  exit 2
fi

api() {  # api METHOD PATH [BODY] -- token stays in the env, out of argv
  local method="$1" path="$2" body="${3:-}"
  if [ -n "$body" ]; then
    curl -sS -X "$method" \
      -H "Authorization: Bearer ${GITHUB_TOKEN}" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      -d "$body" "https://api.github.com${path}"
  else
    curl -sS -X "$method" \
      -H "Authorization: Bearer ${GITHUB_TOKEN}" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com${path}"
  fi
}

echo "==> Checking ${OWNER}/${NAME}"
if api GET "/repos/${OWNER}/${NAME}" | grep -q '"full_name"'; then
  echo "    exists"
else
  if [ "${1:-}" != "--push" ]; then
    echo "    missing (dry run: would create it)"
    echo "==> Would then run: git push -u origin ${BRANCH}"
    exit 0
  fi
  echo "    missing -> creating"
  api POST /user/repos \
    "{\"name\":\"${NAME}\",\"private\":false,\"has_issues\":true,\"has_wiki\":false,\"auto_init\":false}" \
    | grep -o '"full_name": *"[^"]*"' || true
fi

if [ "${1:-}" != "--push" ]; then
  echo "==> Dry run only. Re-run with --push to publish."
  exit 0
fi

echo "==> Pushing ${BRANCH}"
# The credential helper is supplied inline so that .git/config, which the
# sandbox keeps read-only, never needs to be written.
git -c credential.helper='!f() { echo username=x-access-token; echo "password=${GITHUB_TOKEN}"; }; f' \
    push "${REMOTE_URL}" "HEAD:${BRANCH}" --force-with-lease

echo "==> Done: https://github.com/${OWNER}/${NAME}"
