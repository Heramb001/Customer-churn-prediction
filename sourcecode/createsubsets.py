import sys
import time
import subprocess
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Window
from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as F

# function to run linux commands in python
def run_cmd(args_list):
        """
        run linux commands
        """
        # import subprocess
        print('Running system command: {0}'.format(' '.join(args_list)))
        proc = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        s_output, s_err = proc.communicate()
        s_return =  proc.returncode
        return s_return, s_output, s_err 

if __name__ == '__main__':

    start=time.time()   
    #check if file exists and delete the previous files
    (ret, out, err)= run_cmd(['hadoop', 'fs', '-rm', '-r', '-skipTrash', 'project/subsets/*'])
    #create a spark Session
    spark = SparkSession.builder \
    .appName("Sparkify-createSubsetdata") \
    .getOrCreate()
    event_data = "project/data/sparkify_event_data.json"
    
    print('> Starting with PreProcessing...')
    # reading spark Data
    df = spark.read.json(event_data)
    print('> Read Data Successful.')
    print('> Rows: {}, Columns: {}'.format(df.count(), len(df.columns)))
    print('> Starting with creating subsets.')
    subSize = [1087410, 2174820, 4349640, 8699280, 17398560, 25259199]
    for i in range(len(subSize)):
	df1 = df.limit(subSize[i])
    	print('> subset - {} with {} rows.'.format(i,subSize[i]))
    	df1.coalesce(1).write.format('json').save('project/subsets/set'+str(i)+'.json')
	(ret, out, err)= run_cmd(['hadoop', 'fs', '-mv', 'project/subsets/set'+str(i)+'.json/*.json', 'project/data/dataset'+str(i+3)+'.json'])
    spark.stop()
    print('<---- Creation of dataset Complete---->')
    print('<--- Process of creating subsets completed in %i seconds --->'%(time.time() - start))
