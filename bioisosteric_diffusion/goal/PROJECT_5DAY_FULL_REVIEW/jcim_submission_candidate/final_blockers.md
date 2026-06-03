# Final Submission Blockers

Audit date: 2026-06-03.

## Blocker 1: Public V5/Fresh Blind2 tag not verified

**Status:** Open.

**Required before submission:** Push and verify repository tag `v5-fresh-blind2-evidence-lock-20260603` at `https://github.com/user141514/paper1/tree/v5-fresh-blind2-evidence-lock-20260603`.

**What was attempted:** The local branch `codex/jcim-algorithm-archive` and local tag `v5-fresh-blind2-evidence-lock-20260603` were prepared at the current local evidence-lock commit. Repeated `git push` and `git ls-remote --tags origin v5-fresh-blind2-evidence-lock-20260603` attempts failed from the current environment because `github.com:443` could not be reached. `Test-NetConnection github.com -Port 443` also failed. SSH transport to GitHub is network-reachable on ports 22 and 443, but this machine has no usable GitHub SSH key (`Permission denied (publickey)`). Browser-side access to the proposed tag URL returned 404, consistent with the tag not being publicly visible.

**Manuscript implication:** Data Availability now uses submission-facing public-tag wording, but the package cannot be marked submission-ready until the public tag is actually visible in the remote repository.

**Resolution command once network access is available:**

```bash
git push origin codex/jcim-algorithm-archive
git push origin v5-fresh-blind2-evidence-lock-20260603 --force
git ls-remote --tags origin v5-fresh-blind2-evidence-lock-20260603
```

If HTTPS remains blocked, configure a GitHub SSH key for `user141514` on this machine and push using an SSH remote.
