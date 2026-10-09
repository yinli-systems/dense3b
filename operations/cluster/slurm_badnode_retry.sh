#!/bin/bash
# After a training job died with "shards hold no full window" (2026-10-08: some nodes see every JuiceFS shard as empty):
# map the failing ranks to their nodes (one task per node, block order), add them to the job's --exclude list and
# resubmit the same sbatch with the named environment variables, start delayed 1 min; max 8 retries.
#   slurm_badnode_retry.sh SCRIPT LOGFILE VAR...
SCRIPT=$1; LOG=$2; shift 2
R=$(( ${RETRY:-0} + 1 )); [ "$R" -le 8 ] || { echo "RETRY_LIMIT_REACHED"; exit 0; }
bad=$(grep -oE "^\[rank[0-9]+\]: ValueError: shards hold no full window" "$LOG" | grep -oE "rank[0-9]+" | tr -dc "0-9\n" | sort -un)
[ -n "$bad" ] || { echo "NO_DATA_NODE_FAILURE (not resubmitted)"; exit 0; }
hosts=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
EXCL=$(scontrol show job "$SLURM_JOB_ID" | grep -oP "ExcNodeList=\K\S+" || true); [ "$EXCL" = "(null)" ] && EXCL=
for r in $bad; do EXCL="${EXCL:+$EXCL,}${hosts[$r]}"; done
ARGS=(HOME="$HOME" PATH="$PATH" USER="$USER" RETRY="$R")
for v in "$@"; do [ -n "${!v+x}" ] && ARGS+=("$v=${!v}"); done
J=$(env -i "${ARGS[@]}" sbatch --parsable --job-name="$SLURM_JOB_NAME" --exclude="$EXCL" --begin=now+60 "$SCRIPT")
echo "RESUBMITTED $J retry=$R exclude=$EXCL"
