#!/bin/bash

echo '> creating all the directories...'

#hadoop fs -mkdir project
#hadoop fs -mkdir project/data
#hadoop fs -mkdir project/subsets
#:hadoop fs -mkdir project/runs


echo '> downloading data from aws server ...'

wget http://udacity-dsnd.s3.amazonaws.com/sparkify/sparkify_event_data.json

echo '> placing the file in hadoop server'
hadoop fs -mv sparkify_event_data.json project/data/

rm -r sparkify_event_data.json

echo '> files downloaded and placed in hadoop server. Please run createsuubsets.py to create the datasets.'

