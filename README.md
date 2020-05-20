# Customer Churn Prediction

### files

- createsubsets.py  --- python file to create subsets from main data
- preprocess.py     --- python file to preprocess the data using sparksql and rdds
- modelling.py      --- pyspark rdd implementation of logistic regression to model on the preprocessed data
- mliblogistic.py   --- mllib implementation of logistic regression 

- initapp.sh 		--- downloads the data and uploads it to hdfs
- runCode.sh        --- executes the whole pipeline code of preprocessing and model execution.
- setSpark.sh       --- sets Spark home for the server. 

### Instructions on running the Project source code (Do not change anything from below commands as it requires same directory name)

> create project directory in hadoop fs by using below command.

step 1 : ***Create data directories***

hadoop fs -mkdir project

hadoop fs -mkdir project/data

hadoop fs -mkdir project/runs

hadoop fs -mkdir project/subsets

> please upload the data in data/ directory in the hadoop filesystem using below commands.

Step 2 : ***uploading the project data***

hadoop fs -put dataset1.json project/data/

hadoop fs -put dataset2.json project/data/

> run steps 3 and 4 only if you want to execute the code on all the data in order.

step 3 : ***download and place all data in hadoop***

run the below script as it downloads the data from aws server and places it in the hadoop file system.

./initapp.sh

step 4 : ***create subsets***

spark-submit createsubsets.py


step 5 : ***run the entire code***

./runCode.sh <dataset filename> <iterations> <learning Rate>

eg : ./runCode.sh dataset1 100 0.0001

step 6 : ***run mllib version***

spark-submit mliblogistic.py <dataset filename> <iterations>

eg : spark-submit mliblogistic.py project/runs/dataset3/part-00000 100


