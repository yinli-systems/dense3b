# Publication security and provenance

This public repository contains a sanitized copy of the historical NCCL and
Slurm diagnostics. Internal IPv4/link-local addresses were replaced with
explicit placeholders throughout rewritten reachable Git history.

Original and public SHA-256 values for affected archived artifacts are retained
separately in provenance/PUBLIC_NETWORK_REDACTION_AUDIT.json and the updated
inventory records. The published logs are not byte-identical to the original
experiment logs. Numerical experiments were not rerun or replaced.

Original Git bytes are retained only in a private local backup. Keep old clones
from being pushed back to the public repository. GitHub cached commit views,
existing clones, and fork copies may still retain old objects even after a
force push: history rewriting cannot guarantee revocation of previously copied
material. Contact GitHub Support if truly sensitive material must be removed
from GitHub-side caches; support may not purge non-secret infrastructure data.

Historical absolute filesystem paths have been retained as provenance, not as
working credentials. Rewriting them indiscriminately would break archival
receipts and source-specific configurations. Reproducers must configure their
own data and work directories.

To check the publication integrity: python3 scripts/verify_publication.py
