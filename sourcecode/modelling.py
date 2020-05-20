import sys
import numpy as np
from pyspark import SparkContext, SparkConf
import time

np.seterr(all='ignore')

D = 15 # no of features used for modelling

def readPointBatch(iterator):
    strs = list(iterator)
    matrix = np.zeros((len(strs), D + 1))
    for i in range(len(strs)):
        matrix[i] = np.fromstring(strs[i].replace(',', ' '), dtype=np.float64, sep=' ')
    return [matrix]

def gradient(matrix, w,b):
        Y = matrix[:, 0]    # labels (first column of input file)
        X = matrix[:, 1:]   # features
        z = np.dot(X,w)     # calculating X.w
        return (((1.0/(1.0 + np.exp(-Y*z)) - Y) * X.T).sum(1)) # returning gradient

def gradientB(matrix, w,b):
	Y = matrix[:, 0]    # labels (first column of input file)
        X = matrix[:, 1:]   # features
        z = np.dot(X,w)+b     # calculating X.w+b
	return (1.0 / (1.0 + np.exp(-Y * z)) - Y).sum(1)

def add(x, y):
        x += y
        return x

def getMetrics(actual, pred):
    diff = list(2*actual - pred)
    tn = diff.count(0)
    fp = diff.count(-1)
    tp = diff.count(1)
    fn = diff.count(2)
    cm = 'n = '+str(len(pred))+'\n \t\t Actual Positive \t Actual Negative\n Predicted True \t\t '+str(tp)+' \t '+str(fp)+'  \n Predicted False \t\t '+str(fn)+' \t '+str(tn)+' \n'
    return {
        'confusionMatrix' : cm,
        'Precision' : float(tp)/float(tp+fp),
        'Recall' : float(tp)/float(tp+fn),
        'F1-Score' : (2.0 * tp)/(2.0*tp + fp + fn),
        'Accuracy' : float(tp+tn)/float(tp+fp+tn+fn),
        'tpr' : float(tp)/float(tp+fn),
        'fpr' : float(fp)/float(fp+tn),
        }

if __name__ == "__main__":
    
    start_time = time.time()
    conf = SparkConf().set("spark.driver.maxResultSize", "5g")
    sc = SparkContext(conf=conf, appName="logistic-PySpark")
    data = sc.textFile(sys.argv[1])
    # splitting dataset into train and test set
    print('> Splitting the dataset into 80 | 20 ratio.')
    (trainData, testData) = data.randomSplit([0.8, 0.2],seed=42)
    print('> train set : {}, test set : {}'.format(trainData.count(),testData.count()))
    xTrain = trainData.map(lambda line: line[1:])
    yTrain = trainData.map(lambda line: line[0])
    xTrainSC = xTrain.map(lambda line: line.split()).map(lambda value: [float(i) for i in value])
    dataPoints = trainData.mapPartitions(readPointBatch).cache()
    iterations = int(sys.argv[2])
    
    # initializing random weights
    #w = 2 * np.random.ranf(size=D) - 1
    #w = np.random.ranf(size=D)
    w = np.zeros(D)
    b = 0
    # learning rate.
    lr = float(sys.argv[3])
    print('> initital weights : {}'.format(w))
    # for all the iterations
    for i in range(iterations):
        # calculate the gradient and update the weights
        w -= lr * dataPoints.map(lambda m: gradient(m, w, b)).reduce(add)
        # lr - learning rate
        #b -= dataPoints.map(lambda m: gradientB(m, w, b))
        weights_train_data=xTrainSC.map(lambda x: x*w)
        weights_train_data_values=weights_train_data.map(lambda line: sum(line))
        train_pred=weights_train_data_values.map(lambda p: 1/(1+np.exp(-p)))
        #conversion to int train data type
        pred_train_data=train_pred.map(lambda value: int(value))
        label_train_data_values=yTrain.map(lambda value: int(value))
        #conversion of  train rdd to array
        label_v_train_arr = np.array(label_train_data_values.collect())
        pred_v_train_arr = np.array(pred_train_data.collect())
        if i%100 == 0:
		print ('>> On iteration {}, Train data-Accuracy score: {}'.format((i),(label_v_train_arr == pred_v_train_arr).sum().astype(float) / len(pred_v_train_arr)))

    # printing Final Weights
    print("> Final w : " + str(w))
    
    print('> Train Accuracy score: {0}'.format((label_v_train_arr == pred_v_train_arr).sum().astype(float) / len(pred_v_train_arr)))
    
    # training complete, starting on test set
    print('> Starting on TestSet')
    xTest = testData.map(lambda line: line[1:])
    yTest = testData.map(lambda line: line[0])
    xTestScaled = xTest.map(lambda line: line.split()).map(lambda value: [float(i) for i in value])
    weights_testData=xTestScaled.map(lambda x: x*w)
    weights_testData_values=weights_testData.map(lambda line: sum(line))
    testPred=weights_testData_values.map(lambda p: 1/(1+np.exp(-p)))
    
    #conversion to int data type
    pred_v=testPred.map(lambda value: int(value))
    label_v=yTest.map(lambda value: int(value))
    #conversion of rdd to array
    label_v_arr = np.array(label_v.collect())
    pred_v_arr = np.array(pred_v.collect())
    print('> Test Data Accuracy score: {0}'.format((label_v_arr == pred_v_arr).sum().astype(float) / len(pred_v_arr)))
    metrics = getMetrics(label_v_arr, pred_v_arr)
    print('confusion matrix for test data --')
    print(metrics['confusionMatrix'])
    print('F1- Score : {}'.format(metrics['F1-Score']))
    print('<--- Modelling Complete --->')
    print("--- Time of execution for modelling %s seconds ---" % (time.time() - start_time))
    sc.stop()
