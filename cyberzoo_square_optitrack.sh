#!/usr/bin/env bash

# Check whether pwd is project root
if [[ "${PWD##*/}" != crazyflie-suite ]]; then
    echo "Should be run from project root, exiting"
    exit 1
fi

pwd
# Run
python fly_square.py \
    --fileroot data \
    --keywords cyberzoo \
    --logconfig flight/logcfg_snn_pid.json \
    --space flight/space_cyberzoo.yaml \
    --estimator kalman \
    --uwb none \
    --trajectory hover \
    --optitrack state \
    --optitrack_id 2