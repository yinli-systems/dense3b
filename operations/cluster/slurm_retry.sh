#!/bin/bash
# Resubmit a training sbatch whose network preflight failed (2026-10-08): same job name, same --exclude list, the
# named environment variables, start delayed 2 min so the scheduler can hand out a different allocation; max 8 retries.
#   slurm_retry.sh SCRIPT VAR...
R=$(( ${RETRY:-0} + 1 )); [ "$R" -le 8 ] || { echo "RETRY_LIMIT_REACHED"; exit 0; }
EXCL=$(scontrol show job "$SLURM_JOB_ID" | grep -oP "ExcNodeList=\K\S+" || true); [ "$EXCL" = "(null)" ] && EXCL=
ARGS=(HOME="$HOME" PATH="$PATH" USER="$USER" RETRY="$R")
for v in "${@:2}"; do [ -n "${!v+x}" ] && ARGS+=("$v=${!v}"); done
J=$(env -i "${ARGS[@]}" sbatch --parsable --job-name="$SLURM_JOB_NAME" ${EXCL:+--exclude=$EXCL} --begin=now+120 "$1")
echo "RESUBMITTED $J retry=$R exclude=${EXCL:-none}"
