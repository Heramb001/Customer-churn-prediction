import sys
import datetime
import time

import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Window
from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, DoubleType, DateType, FloatType
from pyspark.sql.functions import count, col, udf, desc, max as Fmax, lag, struct, date_add, sum as Fsum,datediff, trunc, row_number, when, coalesce, avg as Favg

def label_churn(x):
    '''
    INPUT
    x: Page
    
    OUTPUT
    Returns 1 if an instance of Churn, else returns 0
    '''
    if x=='Cancellation Confirmation':
        return 1
    elif x=='Downgrade':
        return 1
    else:
        return 0

def compare_date_cols(x,y):
    '''
    Compares x to y. Returns 1 if different
    '''
    if x != y:
        return 0
    else:
        return 1


if __name__ == '__main__':
    
    start=datetime.datetime.now()
    start_time = time.time()
    #create a spark Session
    spark = SparkSession.builder \
    .appName("Sparkify-Preprocessdata") \
    .config("spark.driver.maxResultSize", "5g")\
    .getOrCreate()

    spark.conf.set("spark.sql.debug.maxToStringFields", 100)

    if sys.argv[1] == 'mini':
        event_data = "project/data/mini_sparkify_event_data.json"
    elif sys.argv[1] == 'medium':
        event_data = "project/data/medium_sparkify_event_data.json"
    elif sys.argv[1] == 'large':
        event_data = "project/data/sparkify_event_data.json"
    else:
	event_data = sys.argv[1]
    
    print('> Starting with PreProcessing...')
    # reading spark Data
    df = spark.read.json(event_data)
    print('> Read Data Successful.')
    print('> Rows: {}, Columns: {}'.format(df.count(), len(df.columns)))
    # Dropping the blank userIds
    df = df.where(col('userId')!='')
    print('> Dropping Blank UserIds...')
    print('> Creating Features...')
    # Defining some functions to help pull hour, day, month, and year
    get_hour = udf(lambda x: datetime.datetime.fromtimestamp(x/1000).hour,IntegerType())
    get_day = udf(lambda x: datetime.datetime.fromtimestamp(x/1000).day,IntegerType())
    get_month = udf(lambda x: datetime.datetime.fromtimestamp(x/1000).month,IntegerType())
    get_year = udf(lambda x: datetime.datetime.fromtimestamp(x/1000).year,IntegerType())

    # Creating the columns
    df = df.withColumn('hour',get_hour(col('ts'))) \
        .withColumn('day',get_day(col('ts'))) \
        .withColumn('month',get_month(col('ts'))) \
        .withColumn('year',get_year(col('ts')))
    
    # Also creating a feature with the PySpark DateType() just in case
    get_date = udf(lambda x: datetime.datetime.fromtimestamp(x/1000),DateType())
    df = df.withColumn('date',get_date(col('ts')))
    
    # Creating a column containing 1 if the event was a "NextSong" page visit or 0 otherwise
    listen_flag = udf(lambda x: 1 if x=='NextSong' else 0, IntegerType())
    df = df.withColumn('listen_flag',listen_flag('page'))
    
    # Creating a second table where I will create this feature, then join it back to the main table later
    df_listen_day = df.select(['userId','date','listen_flag']).groupBy(['userId','date']).agg(Fmax('listen_flag')).alias('listen_flag').sort(['userId','date'])
    
    # Defining a window partitioned by User and ordered by date
    window = Window.partitionBy('userId').orderBy(col('date'))
    
    # Using the above defined window and a lag function to create a previous day column
    df_listen_day = df_listen_day.withColumn('prev_day',lag(col('date')).over(window))
    
    # Creating a udf to compare one date to another
    date_group = udf(compare_date_cols, IntegerType())
    
    # Creating another window partitioned by userId and ordered by date
    windowval = (Window.partitionBy('userId').orderBy('date').rangeBetween(Window.unboundedPreceding, 0))
    
    df_listen_day = df_listen_day \
                                .withColumn('date_group',
                                date_group(col('date'), date_add(col('prev_day'),1)) \
                                        # The above line checks if current day and previous day +1 day are equivalent
                                            # If They are equivalent (i.e. consecutive days), return 1
                                ) \
                            .withColumn( \
                                'days_consec_listen',
                                Fsum('date_group').over(windowval)) \
                            .select(['userId','date','days_consec_listen'])
                                        # The above lines calculate a running total summing consecutive listens
    
    # Joining this intermediary table back into the original DataFrame
    df = df.join(other=df_listen_day,on=['userId','date'],how='left')
    print('> computed consecutive_listening_days feature...')    

    # Isolating a few columns and taking the max aggregation to effectively remove duplicates
    df_listen_day = df.select(['userId','date','listen_flag']) \
                        .groupBy(['userId','date']) \
                        .agg(Fmax('listen_flag')).alias('listen_flag').sort(['userId','date'])
                        
    # Re-stating the window
    windowval = Window.partitionBy('userId').orderBy('date')

    # Calculate difference (via datediff) between current date and previous date (taken with lag), and filling na's with 0
    df_last_listen = df_listen_day.withColumn('days_since_last_listen',
                                                datediff(col('date'),lag(col('date')).over(windowval))) \
                                .fillna(0,subset=['days_since_last_listen']) \
                                .select(['userId','date','days_since_last_listen'])

    # Joining back results
    df = df.join(df_last_listen,on=['userId','date'],how='left')
    print('> computed days_since_last_listen feature...')

    # computing listens by month --- not able to execute in turing
   
    # Creating udf's to flag whenever a user visits each particular page
    thU_flag = udf(lambda x: 1 if x=='Thumbs Up' else 0, IntegerType())
    thD_flag = udf(lambda x: 1 if x=='Thumbs Down' else 0, IntegerType())
    err_flag = udf(lambda x: 1 if x=='Error' else 0, IntegerType())
    addP_flag = udf(lambda x: 1 if x=='Add to Playlist' else 0, IntegerType())
    addF_flag = udf(lambda x: 1 if x=='Add Friend' else 0, IntegerType())
    
    # Creating the flag columns
    df = df.withColumn('thU_flag',thU_flag('page')) \
                .withColumn('thD_flag',thD_flag('page')) \
                .withColumn('err_flag',err_flag('page')) \
                .withColumn('addP_flag',addP_flag('page')) \
                .withColumn('addF_flag',addF_flag('page'))    

    # Creating udf
    udf_label_churn = udf(label_churn, IntegerType())
    # Creating column
    df = df.withColumn('Churn',udf_label_churn(col('page')))

    df_listens_user = df.groupBy('userId')\
                .agg(Fmax(col('days_since_last_listen')).alias('most_days_since_last_listen'),
                    Fmax(col('days_consec_listen')).alias('most_days_consec_listen'),
                    Fsum(col('listen_flag')).alias('total_listens'),
                    Fsum(col('thU_flag')).alias('total_thumbsU'),
                    Fsum(col('thD_flag')).alias('total_thumbsD'),
                    Fsum(col('err_flag')).alias('total_err'),
                    Fsum(col('addP_flag')).alias('total_add_pl'),
                    Fsum(col('addF_flag')).alias('total_add_fr')
                    )

    df_sess = df.select(['userId','sessionId','listen_flag','thU_flag','thD_flag','err_flag','addP_flag','addF_flag']) \
                    .groupBy(['userId','sessionId']) \
                    .agg(Fsum(col('listen_flag')).alias('sess_listens'),
                        Fsum(col('thU_flag')).alias('sess_thU'),
                        Fsum(col('thD_flag')).alias('sess_thD'),
                        Fsum(col('err_flag')).alias('sess_err'),
                        Fsum(col('addP_flag')).alias('sess_addP'),
                        Fsum(col('addF_flag')).alias('sess_addF'))

    df_sess_agg = df_sess.groupBy('userId') \
                    .agg(Favg(col('sess_listens')).alias('avg_sess_listens'),
                        Favg(col('sess_thU')).alias('avg_sess_thU'),
                        Favg(col('sess_thD')).alias('avg_sess_thD'),
                        Favg(col('sess_err')).alias('avg_sess_err'),
                        Favg(col('sess_addP')).alias('avg_sess_addP'),
                        Favg(col('sess_addF')).alias('avg_sess_addF'))

    print('> computed Thumbs Up, Thumbs Down, Error, Add to playlist, Add Friend from session data...')
    dfUserMatrix = df.groupBy('userId').agg(Fmax(col('gender')).alias('gender')
                                                ,Fmax(col('churn')).alias('churn'))

    dfUserMatrix = dfUserMatrix.join(df_listens_user,['userId']).join(df_sess_agg,['userId'])
    #check if data has any null in it
    #for i in dfUserMatrix.columns:
    #	print('> validating null values in column {} : {}'.format(i,dfUserMatrix.where(col(i).isNull()).count())) 
    for i in dfUserMatrix.columns:
    	dfUserMatrix = dfUserMatrix.where(col(i).isNotNull())
    print('> created final dataframe after preprocessing...')
    
    #gender_indexer = StringIndexer(inputCol='gender',outputCol='gender_indexed')
    #gender_indexer.setHandleInvalid('skip')
    #fitted_gender_indexer = gender_indexer.fit(dfUserMatrix)
    #dfModel = fitted_gender_indexer.transform(dfUserMatrix)
    
    features = [col for col in dfUserMatrix.columns if col not in ('userId','gender','churn')]
    print('> features computed...')
    print('> features are : {}'.format(str(features)))
    #print('> Final Data has Rows: {}, Columns: {}'.format(len(dfUserMatrix.collect()), len(dfUserMatrix.columns)))
    rdd = dfUserMatrix.rdd.map(tuple)
    rdd2 = rdd.map(lambda elm: str(elm[2]) + " " + str(elm[3]) + " " + str(elm[4]) + " " + str(elm[5]) + " " + str(elm[6]) + " " 
                + str(elm[7]) + " " + str(elm[8]) + " " + str(elm[9]) + " " + str(elm[10]) + " " + str(elm[11]) + " " 
                + str(elm[12]) + " " + str(elm[13]) + " " + str(elm[14]) + " " + str(elm[15]) + " " + str(elm[16]) + " "
                + str(1 if elm[1] == 'F' else 0))
    print('> final rdd has {} Rows'.format(rdd2.count()))
    rdd2.coalesce(1).saveAsTextFile(sys.argv[2])
    print('> saved in hadoop fs path : {}'.format(str(sys.argv[2])))
    spark.stop()
    print("--- Time of Execution for preprocessing : %s seconds ---" % (time.time() - start_time))
