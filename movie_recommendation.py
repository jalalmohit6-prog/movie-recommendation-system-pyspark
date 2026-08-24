from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# --------------------------------------------------
# 1. Spark Session
# --------------------------------------------------

spark = SparkSession.builder \
    .appName("Movie Recommendation - ALS") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# --------------------------------------------------
# 2. HDFS Paths
# --------------------------------------------------

movies_path = "hdfs://localhost:9000/movie_recommendation/movies/movies.csv"

ratings_path = "hdfs://localhost:9000/movie_recommendation/ratings/ratings.csv"


# --------------------------------------------------
# 3. Read Movies
# --------------------------------------------------

movies = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv(movies_path)


# --------------------------------------------------
# 4. Read Ratings
# --------------------------------------------------

ratings = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv(ratings_path)


# --------------------------------------------------
# 5. Clean Ratings
# --------------------------------------------------

ratings = ratings.dropDuplicates(
    ["userId", "movieId"]
)

ratings = ratings.dropna(
    subset=[
        "userId",
        "movieId",
        "rating"
    ]
)


print("\n========== CLEAN DATA ==========")

print("Movies:", movies.count())
print("Ratings:", ratings.count())


# --------------------------------------------------
# 6. Train / Test Split
# --------------------------------------------------

train, test = ratings.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("\n========== TRAIN TEST SPLIT ==========")

print("Training records:", train.count())
print("Testing records:", test.count())


# --------------------------------------------------
# 7. ALS MODEL
# --------------------------------------------------

als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",

    rank=10,

    maxIter=10,

    regParam=0.1,

    nonnegative=True,

    implicitPrefs=False,

    coldStartStrategy="drop"
)


# --------------------------------------------------
# 8. Train Model
# --------------------------------------------------

print("\n========== TRAINING ALS MODEL ==========")

model = als.fit(train)

print("ALS model trained successfully!")


# --------------------------------------------------
# 9. Prediction
# --------------------------------------------------

predictions = model.transform(test)


print("\n========== PREDICTIONS ==========")

predictions.show(
    20,
    truncate=False
)


# --------------------------------------------------
# 10. RMSE Evaluation
# --------------------------------------------------

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(
    predictions
)


print("\n========== MODEL PERFORMANCE ==========")

print("RMSE:", rmse)


# --------------------------------------------------
# 11. Generate Recommendations
# --------------------------------------------------

print("\n========== USER RECOMMENDATIONS ==========")

user_recommendations = model.recommendForAllUsers(
    5
)

user_recommendations.show(
    truncate=False
)


# --------------------------------------------------
# 12. Convert Recommendation Array
# --------------------------------------------------

recommendations = user_recommendations \
    .withColumn(
        "recommendation",
        F.explode("recommendations")
    )


recommendations = recommendations.select(
    "userId",

    F.col(
        "recommendation.movieId"
    ).alias("movieId"),

    F.col(
        "recommendation.rating"
    ).alias("predicted_rating")
)


# --------------------------------------------------
# 13. Join Movie Names
# --------------------------------------------------

final_recommendations = recommendations.join(
    movies,
    on="movieId",
    how="left"
)


# --------------------------------------------------
# 14. Final Recommendations
# --------------------------------------------------

print("\n========== FINAL MOVIE RECOMMENDATIONS ==========")

final_recommendations.select(
    "userId",
    "movieId",
    "title",
    "genres",
    "predicted_rating"
).orderBy(
    "userId",
    F.desc("predicted_rating")
).show(
    50,
    truncate=False
)


# --------------------------------------------------
# 15. Save Recommendations to HDFS
# --------------------------------------------------

output_path = \
    "hdfs://localhost:9000/movie_recommendation/als_output"


final_recommendations.select(
    "userId",
    "movieId",
    "title",
    "genres",
    "predicted_rating"
).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)


print("\nRecommendations saved to HDFS:")
print(output_path)


# --------------------------------------------------
# 16. Stop Spark
# --------------------------------------------------

spark.stop()