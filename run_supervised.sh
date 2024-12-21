#!/bin/bash

accelerate launch --config_file train_configs/mntp/az_zero2.yaml \
    experiments/run_supervised.py \
    train_configs/supervised/MetaLlama3.1.json
