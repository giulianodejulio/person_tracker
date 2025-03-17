#!/bin/bash

# Variables for configurability
MODEL="lstm"          # Change to 'lstm', 'sgan', 'vae'
SUFFIX="social"      # Change to 'vanilla', 'occupancy', 'directional', 'social', 'hiddenstatemlp', 'nn', 'attentionmlp', 'nn_lstm', 'traj_pool' 
SCENE_ID="2158"

# Generate controlled trajectories
cd $WORKSPACE_TO_SOURCE/src/hrii_person_tracker/trajnetpp_test/trajnetplusplusbaselines/
mkdir -p data/raw/controlled
python3 -m trajnetdataset.controlled_data --simulator 'orca' --num_ped 5 --num_scenes 1000

# Convert the data
python3 -m trajnetdataset.convert --linear_threshold 0.3 --acceptance 0 0 1.0 0 --synthetic

# Move the converted data and goal files to the appropriate directories
mv output ../trajnetplusplusbaselines/DATA_BLOCK/synth_data
mv goal_files/ ../trajnetplusplusbaselines/

# Train the model
cd ../trajnetplusplusbaselines/
python3 -m trajnetbaselines.$MODEL.trainer --path synth_data --type ${SUFFIX}

# Evaluate the model
OUTPUT_DIR="OUTPUT_BLOCK/synth_data"
PKL_FILE="$OUTPUT_DIR/${MODEL}_${SUFFIX}_None.pkl"
python3 -m trajnetbaselines.$MODEL.trajnet_evaluator --path synth_data --output $PKL_FILE

# Visualize predictions
python3 -m evaluator.visualize_predictions \
  DATA_BLOCK/synth_data/test_private/orca_five_synth.ndjson \
  DATA_BLOCK/synth_data/test_pred/"${MODEL}_${SUFFIX}_None_modes1/orca_five_synth.ndjson" \
  --n 10 --id $SCENE_ID

