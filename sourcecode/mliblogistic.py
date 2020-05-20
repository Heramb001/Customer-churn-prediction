
import pyspark
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext, SparkConf

import sys
import numpy as np
import time

# Load and parse the data
def parsePoint(line):
    	values = [float(x) for x in line.split(' ')]
    	return LabeledPoint(values[-1], values[:-1])

if __name__ == "__main__":
	print('> Implementing MLLIB model on data')
	start = time.time()
        conf = SparkConf().set("spark.driver.maxResultSize", "5g")
	sc = SparkContext(conf=conf, appName="logistic-mllib")
	data = sc.textFile(sys.argv[1])
	# splitting dataset into train and test set
    	print('> Splitting the dataset into 80 | 20 ratio.')
    	(trainData, testData) = data.randomSplit([0.8, 0.2],seed=42)
	trainDt = trainData.map(parsePoint)
	iterations = int(sys.argv[2])
	for m in range(iterations):
		model = LogisticRegressionWithLBFGS.train(trainDt,m)
	print("> final weights :",str(model.weights))
	print("> length of weights :",len(model.weights))
	# Evaluating the model on training data
	labelsAndPreds = trainDt.map(lambda p: (float(model.predict(p.features)), p.label))
	trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(trainDt.count())
	print("> Training Error = " + str(trainErr))
	metrics = MulticlassMetrics(labelsAndPreds)
	print('> Training Accuracy : {0}'.format(metrics.accuracy))
	
	# applying on test set
	testDt = testData.map(parsePoint)
	testPreds = testDt.map(lambda p: (float(model.predict(p.features)), p.label))
	metrics = MulticlassMetrics(testPreds)
	print('> Test Accuracy : {}'.format(metrics.accuracy))
	print('> Confusion Matrix')
	print(metrics.confusionMatrix().toArray())
	print('> F1-Score : {}'.format(metrics.fMeasure()))
	print('<--- Complete --->')
	print("--- Time of execution for mllib version %s seconds ---" % (time.time() - start))
	sc.stop()
	print('')
