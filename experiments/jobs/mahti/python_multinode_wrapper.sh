#!/bin/bash

nvidia-smi dmon -s mu -d 5 -o TD > "${LOG_DIR}/gpu_load-${EXP_ID}-${PPID}.log" &
singularity_wrapper exec --nv "${SAND_BOX}" python -u "$@"
