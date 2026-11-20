#!/usr/bin/env bash
# Third training seed (--seed 2) of the main models: same job as run_seed.sh.
SEED=2 exec bash "$(dirname "$0")/run_seed.sh"
