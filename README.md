# recommender
Spark ML collaborative filtering application to generate recommendations

Blog post: https://www.schaper.io/blog/2017/10/30/building-a-recommendation-engine-using-spark-and-aws-emr

## Prerequisites
* Apache Spark = 2.2.0
* Python >= 2.7

## Run
```
$SPARK_HOME/bin/spark-submit --master spark://master-url:7077 \
  recommender.py \
  --input path/to/inputfile.csv \
  --output path/to/outputdir
```
