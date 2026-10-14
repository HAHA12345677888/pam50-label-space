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
NAME="${GITHUB_REPO:-pam50-label-space}"
BRANCH="${GITHUB_BRANCH:-main}"
REMOTE_URL="https://github.com/${OWNER}/${NAME}.git"

# NOTE: the local 'origin' still points at the old repository
# (git@github.com:HAHA1122344/breast-cancer-multimodal-transformer.git). This script
# deliberately pushes to REMOTE_URL by explicit URL rather than to 'origin', so it does not
# need .git/config to be writable. To repoint 'origin' for your own use, run this yourself
# from your own terminal:
#   git remote set-url origin https://github.com/HAHA12345677888/pam50-label-space.git

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
status="$(curl -sS -o /dev/null -w '%{http_code}' -K - "https://api.github.com/repos/${OWNER}/${NAME}" \
  <<< "$(printf 'header = "Authorization: Bearer %s"\n' "${GITHUB_TOKEN}")")"
case "$status" in
  200) echo "    exists" ;;
  404)
    if [ "${1:-}" != "--push" ]; then
      echo "    missing (dry run: would create it)"
      echo "==> Would then run: git push ${REMOTE_URL} HEAD:${BRANCH}"
      exit 0
    fi
    echo "    missing -> creating"
    api POST /user/repos \
      "{\"name\":\"${NAME}\",\"private\":false,\"has_issues\":true,\"has_wiki\":false,\"auto_init\":false}" \
      | grep -o '"full_name": *"[^"]*"' || true
    ;;
  *)
    echo "    unexpected HTTP ${status} from the API — refusing to proceed."
    echo "    A 401/403 means the token is bad or lacks scope; anything else is worth a look."
    exit 3
    ;;
esac

if [ "${1:-}" != "--push" ]; then
  echo "==> Dry run only. Re-run with --push to publish."
  exit 0
fi

echo "==> Pushing ${BRANCH}"
# No --force here, deliberately. The intended target is a fresh repository that only ever
# receives this history, so a fast-forward push is always correct, and --force-with-lease
# cannot be relied on when pushing to a bare URL (there is no remote-tracking ref to compare
# against). If the push is ever rejected as non-fast-forward, something changed on the remote
# that this script did not put there; that is worth reading, not overwriting.
# The credential helper is supplied inline so that .git/config, which the
# sandbox keeps read-only, never needs to be written.
if ! git -c credential.helper='!f() { echo username=x-access-token; echo "password=${GITHUB_TOKEN}"; }; f' \
    push "${REMOTE_URL}" "HEAD:${BRANCH}"; then
  echo
  echo "==> Push failed. If the message above says 'non-fast-forward' or 'rejected', the remote"
  echo "    has commits this history does not contain. Inspect before forcing anything:"
  echo "      git ls-remote ${REMOTE_URL}"
  exit 4
fi

echo "==> Done: https://github.com/${OWNER}/${NAME}"
