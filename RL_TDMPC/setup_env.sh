#!/usr/bin/env bash
# Source this file after activating the steve_tdmpc Conda environment:
#   source RL_TDMPC/setup_env.sh

if [[ "${CONDA_DEFAULT_ENV:-}" != "steve_tdmpc" ]]; then
    echo "Warning: expected Conda environment 'steve_tdmpc', current: '${CONDA_DEFAULT_ENV:-none}'" >&2
fi

# Do not retain paths from the original Python 3.8 SOFA installation. Mixing
# SOFA core libraries and BeamAdapter builds produces ABI/undefined-symbol errors.
unset SOFA_PLUGIN_PATH
unset PYTHONPATH
unset LD_LIBRARY_PATH

export SOFA_ROOT="${STEVE_SOFA_ROOT:-/home/bizon/SOFA_TDMPC/install}"
export SOFAPYTHON3_ROOT="$SOFA_ROOT/plugins/SofaPython3"
export PYTHONPATH="$SOFAPYTHON3_ROOT/lib/python3/site-packages"
export LD_LIBRARY_PATH="$SOFA_ROOT/lib:$SOFAPYTHON3_ROOT/lib:$SOFA_ROOT/plugins/BeamAdapter/lib"

if [[ ! -f "$SOFA_ROOT/plugins/BeamAdapter/lib/libBeamAdapter.so" ]]; then
    echo "Error: BeamAdapter was not found under SOFA_ROOT=$SOFA_ROOT" >&2
    return 1 2>/dev/null || exit 1
fi

echo "TD-MPC2 environment ready: $SOFA_ROOT"
