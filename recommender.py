import sys
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.sql import Row
from pyspark.sql.functions import explode

# Example Spark application to generate recommendations using collaborative
# filtering and implicit feedback.
#
# USAGE:
# spark-submit --master spark://master-url:7077 recommender.py --data path/to/data.csv

# Create a spark session
spark = SparkSession.builder.appName('Recommender').getOrCreate()

# Check presence of data filepath argument
args = dict(zip(['command', '--input', 'inputFile', '--output', 'outputFile'], sys.argv))
if not 'inputFile' in args:
    raise Exception('Expected --input arg with input file path')

if not 'outputFile' in args:
    raise Exception('Expected --output arg with output directory path')

# Read data from a CSV file and create a DataFrame with rows
lines   = spark.read.text(args['inputFile']).rdd
parts   = lines.map(lambda row: row.value.split(','))
records = parts.map(lambda p: Row(userId=str(p[0]),
                                  productId=str(p[1]),
                                  rating=int(p[2])))

ratings = spark.createDataFrame(records)

# Convert UUID strings into indices. This step isn't needed if your values
# are already integers or doubles.
userIndexer = StringIndexer(inputCol='userId', outputCol='userIndex').fit(ratings)
productIndexer = StringIndexer(inputCol='productId', outputCol='productIndex').fit(ratings)

pipeline = Pipeline(stages=[userIndexer, productIndexer])
indexedRatings = pipeline.fit(ratings).transform(ratings)

# Randomly split up the data and use 80% for training and the other 20%
# for testing and evaluation.
(training, test) = indexedRatings.randomSplit([0.8, 0.2])

regParams = [0.01, 0.1]
ranks = [16]
alphas = [10.0, 20.0, 40.0, 60.0, 80.0]

# Train the model to find the best RMSE
for regParam in regParams:
    for rank in ranks:
        for alpha in alphas:
            als = ALS(maxIter=10,
                      regParam=regParam,
                      rank=rank,
                      alpha=alpha,
                      seed=8427,
                      implicitPrefs=True,
                      userCol='userIndex',
                      itemCol='productIndex',
                      ratingCol='rating',
                      coldStartStrategy='drop')

            model = als.fit(training)
            predictions = model.transform(test)
            evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
            rmse = evaluator.evaluate(predictions)

            print('For regParam: ' + str(regParam) + ', rank: ' +
                  str(rank) + ', alpha: ' + str(alpha) + ', RMSE: ' + str(rmse))

# Get ten recommendations per user
userRecs = model.recommendForAllUsers(10)

# Transform the recommendations into a flat structure with the columns
# | userIndex | productIndex | rating |
flatUserRecs = userRecs.withColumn('productAndRating', explode(userRecs.recommendations)) \
                       .select('userIndex', 'productAndRating.*')

# Convert the indices we created earlier back into UUID strings. This step isn't
# needed if your values didn't need to be encoded with StringIndexer.
userConverter = IndexToString(inputCol='userIndex', outputCol='userId', labels=userIndexer.labels)
productConverter = IndexToString(inputCol='productIndex', outputCol='productId', labels=productIndexer.labels)

convertedUserRecs = Pipeline(stages=[userConverter, productConverter]).fit(indexedRatings).transform(flatUserRecs)

# Save our results into CSV files under the specified output directory
convertedUserRecs.write.csv(args['outputFile'])
