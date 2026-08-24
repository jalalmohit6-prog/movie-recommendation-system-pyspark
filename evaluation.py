from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


# ==================================================
# 1. CREATE SPARK SESSION
# ==================================================

spark = SparkSession.builder \
    .appName("Movie Recommendation Evaluation") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# ==================================================
# 2. HDFS PATH
# ==================================================

ratings_path = \
    "hdfs://localhost:9000/movie_recommendation/ratings/ratings.csv"


# ==================================================
# 3. READ RATINGS
# ==================================================

ratings = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv(ratings_path)


# Clean data

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


print("\n========== DATA ==========")

print("Total Ratings:", ratings.count())


# ==================================================
# 4. TRAIN / TEST SPLIT
# ==================================================

train, test = ratings.randomSplit(
    [0.8, 0.2],
    seed=42
)


print("\n========== TRAIN TEST ==========")

print("Training:", train.count())

print("Testing:", test.count())


# ==================================================
# 5. TRAIN ALS
# ==================================================

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


model = als.fit(train)


print("\nALS model trained.")


# ==================================================
# 6. PREDICTIONS
# ==================================================

predictions = model.transform(test)


# ==================================================
# 7. RMSE
# ==================================================

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)


rmse = evaluator.evaluate(
    predictions
)


print("\n========== RMSE ==========")

print("RMSE:", rmse)


# ==================================================
# 8. CREATE TOP 5 RECOMMENDATIONS
# ==================================================

recommendations = model.recommendForAllUsers(
    5
)


recommendations = recommendations \
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


# ==================================================
# 9. RELEVANT MOVIES
# ==================================================

# Rating >= 4 is considered relevant

relevant = test.filter(
    F.col("rating") >= 4
).select(
    "userId",
    "movieId"
).dropDuplicates()


print("\n========== RELEVANT MOVIES ==========")

print(
    "Relevant movies:",
    relevant.count()
)


# ==================================================
# 10. FIND CORRECT RECOMMENDATIONS
# ==================================================

correct = recommendations.join(
    relevant,
    ["userId", "movieId"],
    "inner"
)


true_positive = correct.count()


# ==================================================
# 11. TOTAL RECOMMENDATIONS
# ==================================================

total_recommendations = recommendations.count()


# ==================================================
# 12. TOTAL RELEVANT MOVIES
# ==================================================

total_relevant = relevant.count()


# ==================================================
# 13. PRECISION
# ==================================================

if total_recommendations > 0:

    precision = (
        true_positive /
        total_recommendations
    )

else:

    precision = 0


# ==================================================
# 14. RECALL
# ==================================================

if total_relevant > 0:

    recall = (
        true_positive /
        total_relevant
    )

else:

    recall = 0


# ==================================================
# 15. F1 SCORE
# ==================================================

if precision + recall > 0:

    f1 = (
        2 * precision * recall
    ) / (
        precision + recall
    )

else:

    f1 = 0


# ==================================================
# 16. DISPLAY RESULTS
# ==================================================

print("\n========================================")
print("MOVIE RECOMMENDATION PERFORMANCE")
print("========================================")

print(
    "RMSE      :",
    rmse
)

print(
    "Precision :",
    precision
)

print(
    "Recall    :",
    recall
)

print(
    "F1 Score  :",
    f1
)

print("========================================")


# ==================================================
# 17. STOP SPARK
# ==================================================

spark.stop()