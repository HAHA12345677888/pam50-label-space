#!/usr/bin/env python3
"""Publish the committed tree of this repository to GitHub through the REST API.

Why this exists
---------------
`git push` needs an authenticated HTTPS connection to github.com. In the managed
sandbox github.com is not reachable (curl returns 000 and git reports
"CONNECT tunnel failed, response 502"), while api.github.com is reachable and
authenticated. So publication goes through the REST API instead.

Because the API builds the commit itself, it stamps its own committer, date and
parent, and the resulting SHA is NOT the same as the local one even when the
content is identical. That makes `git status -sb` and `git rev-parse HEAD`
useless for answering "is the remote up to date?" — they compare against a local
`origin` ref that the API never updates.

So this script never reasons about commits. It compares CONTENT: every path in
the local `HEAD` tree against every path in the remote tree, by blob SHA (git
object IDs are content-addressed, so equal SHA means equal bytes). It uploads
what differs, deletes what is gone, and says so. It is therefore safe to run
repeatedly and safe when local and remote histories have diverged.

Usage
-----
    GITHUB_TOKEN=... python3 scripts/publish_via_api.py [--dry-run] [--repo owner/name]

Set PUBLISH_REPO / PUBLISH_BRANCH to override the defaults. The token is read
from GITHUB_TOKEN (falling back to GH_PUSH_TOKEN); it is never logged.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

DEFAULT_REPO = "HAHA12345677888/pam50-label-space"
DEFAULT_BRANCH = "main"
API = "https://api.github.com"


def gh(repo: str, token: str, method: str, path: str, payload=None):
    req = urllib.request.Request(
        f"{API}/repos/{repo}{path}",
        method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "publish-via-api/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path} -> HTTP {e.code}: {e.read().decode()[:300]}")


def git(root: str, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout


def local_tree(root: str) -> dict:
    """{path: (mode, blob_sha)} for everything committed at HEAD."""
    out = {}
    for line in git(root, "ls-tree", "-r", "HEAD").splitlines():
        meta, path = line.split("\t", 1)
        mode, _type, sha = meta.split()
        out[path] = (mode, sha)
    return out


def remote_tree(repo: str, token: str, branch: str):
    """({path: (mode, sha)}, commit_sha) for the remote branch tip."""
    commit = gh(repo, token, "GET", f"/git/ref/heads/{branch}")["object"]["sha"]
    tree_sha = gh(repo, token, "GET", f"/git/commits/{commit}")["tree"]["sha"]
    payload = gh(repo, token, "GET", f"/git/trees/{tree_sha}?recursive=1")
    if payload.get("truncated"):
        raise SystemExit("remote tree listing was truncated; refusing to guess")
    out = {}
    for t in payload["tree"]:
        if t["type"] == "blob":
            out[t["path"]] = (t["mode"], t["sha"])
    return out, commit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("PUBLISH_REPO", DEFAULT_REPO))
    ap.add_argument("--branch", default=os.environ.get("PUBLISH_BRANCH", DEFAULT_BRANCH))
    ap.add_argument("--root", default=".")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--message", default=None)
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_PUSH_TOKEN")
    if not token:
        print("no token in GITHUB_TOKEN / GH_PUSH_TOKEN", file=sys.stderr)
        return 1

    loc = local_tree(args.root)
    rem, rem_commit = remote_tree(args.repo, token, args.branch)
    print(f"repo        {args.repo}  branch {args.branch}")
    print(f"remote tip  {rem_commit[:12]}   {len(rem)} files")
    print(f"local HEAD  {git(args.root, 'rev-parse', 'HEAD').strip()[:12]}   {len(loc)} files")

    added = sorted(p for p in loc if p not in rem)
    changed = sorted(p for p in loc if p in rem and rem[p][1] != loc[p][1])
    removed = sorted(p for p in rem if p not in loc)

    if not (added or changed or removed):
        print("\nIN SYNC — remote content already matches HEAD. Nothing to publish.")
        return 0

    for p in added:
        print(f"  + {p}")
    for p in changed:
        print(f"  ~ {p}")
    for p in removed:
        print(f"  - {p}")
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    entries = []
    for path in added + changed:
        mode, sha = loc[path]
        if mode == "120000":
            data = git(args.root, "cat-file", "blob", sha).encode()
        else:
            data = subprocess.run(
                ["git", "cat-file", "blob", sha], cwd=args.root,
                capture_output=True, check=True).stdout
        got = gh(args.repo, token, "POST", "/git/blobs",
                 {"content": base64.b64encode(data).decode(), "encoding": "base64"})
        entries.append({"path": path, "mode": mode, "type": "blob", "sha": got["sha"]})
    for path in removed:
        entries.append({"path": path, "mode": "100644", "type": "blob", "sha": None})

    base_tree = gh(args.repo, token, "GET", f"/git/commits/{rem_commit}")["tree"]["sha"]
    new_tree = gh(args.repo, token, "POST", "/git/trees",
                  {"base_tree": base_tree, "tree": entries})["sha"]
    message = args.message or git(args.root, "log", "-1", "--pretty=%B").strip()
    if not message:
        message = "publish: sync committed tree"
    commit = gh(args.repo, token, "POST", "/git/commits",
                {"message": message, "tree": new_tree, "parents": [rem_commit]})
    gh(args.repo, token, "PATCH", f"/git/refs/heads/{args.branch}",
       {"sha": commit["sha"], "force": False})
    print(f"\npublished {commit['sha'][:12]} "
          f"({len(added)} added, {len(changed)} changed, {len(removed)} removed)")
    print("Note: local and remote SHAs will differ by construction. Judge sync by "
          "re-running this script, not by `git status`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
