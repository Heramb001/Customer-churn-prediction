#!/bin/bash

./setSpark.sh

# remove all the previously generated files
hadoop fs -rm -r project/runs/${1}/
# preprocessing
spark-submit preprocess.py project/data/${1}.json project/runs/${1}
#modelling and metrics
spark-submit modelling.py project/runs/${1}/part-00000 ${2} ${3}
