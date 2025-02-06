#!/bin/bash
WORK_DIR="/scratch/zzs5287/poison_data_valuation"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
echo $PYTHONPATH



python attack/privacy_attack/run.py --dataset fitzpatrick


