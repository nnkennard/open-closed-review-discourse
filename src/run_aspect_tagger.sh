#!/bin/bash
echo "\nInstalling dependencies for the aspect tagger ...";
cd "../ReviewAdvisor/tagger"
pip install -r 'requirements.txt'


echo "\nRunning the aspect tagger ...";
sh run.sh $1

cd "../../src/"