#!/bin/bash

##################################################
# Step 1: Process annotations
##################################################
echo "\nProcessing annotations ...";
python process_labeled_data.py



##################################################
# Step 2: Run models and generate features
##################################################
echo "\nRunning models and generating features"
python generate_argument_labels.py -m 'scibert'

python generate_aspect_tagger_input_data.py
sh run_aspect_tagger.sh "iclr19 neurips19 iclr18 neurips18"

# TODO: Add specificity data generation and model running code
python generate_specificity_labels.py

# TODO: Add corpus creation code for any JSON
for YEAR in {2018..2021}; do
    python create_iclr_json.py -y $YEAR;
    python create_iclr_corpus.py -y $YEAR;
    python add_politeness_features.py -c 'iclr' -y $YEAR;
done
python create_neurips_corpus.py
for YEAR in {2018..2019}; do
    python add_politeness_features.py -c 'neurips' -y $YEAR;
done


##################################################
# Step 3: Get all features into a single csv file
##################################################
echo "\nLoading features for labeled data...";
python load_features.py -l True

# echo "\nLoading features for unlabeled data...";
# declare -a arr=("iclr19" "neurips19" "neurips19_sm" "iclr18" "neurips18" "neurips18_sm")
# for VENUE in "${arr[@]}"; do
#    echo "$VENUE";
#    python load_features.py -u -v $VENUE;
# done


##################################################
# Step 4: Find significant features
##################################################
echo "\nFinding significant features ...";
python analyze_features.py

echo "\nComputing feature importance ...";
python feature_importance.py
