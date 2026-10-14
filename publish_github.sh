#!/usr/bin/env bash
# Create (if missing) and push this repository to GitHub.
#
# The token is read from the environment. It is never printed and never written
# to a file. For the API calls it is handed to curl through a config stream on
# stdin (-K -), so it does not appear in curl's argument vector and is not
# visible to ps or /proc/<pid>/cmdline. The git push reaches GitHub over HTTPS
# with the token echoed to git through a pipe by an inline credential helper,
# for the same reason. Note that the token does live in this process's
# environment, which is readable by processes running as the same user; that is
# unavoidable for any credential the session injects.
# Create the credential once under Customize -> Credentials -> Add Credential;
# it then arrives as GITHUB_TOKEN in every session.
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

api() {  # api METHOD PATH [BODY] -- token goes to curl via stdin, never argv
  local method="$1" path="$2" body="${3:-}"
  {
    printf 'header = "Authorization: Bearer %s"\n' "${GITHUB_TOKEN}"
    printf 'header = "Accept: application/vnd.github+json"\n'
    printf 'header = "X-GitHub-Api-Version: 2022-11-28"\n'
    if [ -n "$body" ]; then
      printf 'data = "%s"\n' "$(printf '%s' "$body" | sed 's/\\/\\\\/g; s/"/\\"/g')"
    fi
  } | curl -sS -X "$method" -K - "https://api.github.com${path}"
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
