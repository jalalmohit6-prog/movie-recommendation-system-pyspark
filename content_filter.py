from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import IDF
from pyspark.ml.feature import Normalizer


# --------------------------------------------------
# 1. Create Spark Session
# --------------------------------------------------

spark = SparkSession.builder \
    .appName("Content Based Movie Recommendation") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# --------------------------------------------------
# 2. HDFS Movie Path
# --------------------------------------------------

movies_path = \
    "hdfs://localhost:9000/movie_recommendation/movies/movies.csv"


# --------------------------------------------------
# 3. Read Movies
# --------------------------------------------------

movies = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv(movies_path)


print("\n========== MOVIES ==========")

movies.show(
    10,
    truncate=False
)


# --------------------------------------------------
# 4. Handle Missing Values
# --------------------------------------------------

movies = movies.fillna({
    "genres": "",
    "director": "",
    "actors": ""
})


# --------------------------------------------------
# 5. Combine Movie Features
# --------------------------------------------------

movies = movies.withColumn(
    "features",
    F.concat_ws(
        " ",
        F.col("genres"),
        F.col("director"),
        F.col("actors")
    )
)


print("\n========== MOVIE FEATURES ==========")

movies.select(
    "movieId",
    "title",
    "features"
).show(
    10,
    truncate=False
)


# --------------------------------------------------
# 6. Tokenization
# --------------------------------------------------

tokenizer = Tokenizer(
    inputCol="features",
    outputCol="words"
)

words_data = tokenizer.transform(
    movies
)


# --------------------------------------------------
# 7. TF-IDF
# --------------------------------------------------

hashing_tf = HashingTF(
    inputCol="words",
    outputCol="rawFeatures",
    numFeatures=1000
)

featurized_data = hashing_tf.transform(
    words_data
)


idf = IDF(
    inputCol="rawFeatures",
    outputCol="tfidf"
)

idf_model = idf.fit(
    featurized_data
)

movies_features = idf_model.transform(
    featurized_data
)


# --------------------------------------------------
# 8. Normalize Vectors
# --------------------------------------------------

normalizer = Normalizer(
    inputCol="tfidf",
    outputCol="featuresVector",
    p=2.0
)

movies_features = normalizer.transform(
    movies_features
)


# --------------------------------------------------
# 9. Select Movie
# --------------------------------------------------

selected_movie_id = 1


selected_movie = movies_features.filter(
    F.col("movieId") == selected_movie_id
).first()


selected_vector = selected_movie["featuresVector"]

selected_title = selected_movie["title"]


print("\n======================================")
print("Selected Movie:", selected_title)
print("Movie ID:", selected_movie_id)
print("======================================")


# --------------------------------------------------
# 10. Cosine Similarity Function
# --------------------------------------------------

def cosine_similarity(v1, v2):

    dot_product = float(
        v1.dot(v2)
    )

    norm1 = float(
        v1.norm(2)
    )

    norm2 = float(
        v2.norm(2)
    )

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (
        norm1 * norm2
    )


similarity_udf = F.udf(
    lambda vector:
        cosine_similarity(
            selected_vector,
            vector
        ),
    DoubleType()
)


# --------------------------------------------------
# 11. Generate Similar Movies
# --------------------------------------------------

recommendations = movies_features \
    .withColumn(
        "similarity",
        similarity_udf(
            F.col("featuresVector")
        )
    ) \
    .filter(
        F.col("movieId") != selected_movie_id
    ) \
    .orderBy(
        F.desc("similarity")
    )


# --------------------------------------------------
# 12. Display Recommendations
# --------------------------------------------------

print(
    "\n========== CONTENT BASED RECOMMENDATIONS =========="
)

recommendations.select(
    "movieId",
    "title",
    "genres",
    "director",
    "similarity"
).show(
    10,
    truncate=False
)


# --------------------------------------------------
# 13. Save Content Recommendations
# --------------------------------------------------

output_path = \
    "hdfs://localhost:9000/movie_recommendation/content_output"


recommendations.select(
    "movieId",
    "title",
    "genres",
    "director",
    "similarity"
).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)


print(
    "\nContent recommendations saved to HDFS:"
)

print(output_path)


# --------------------------------------------------
# 14. Stop Spark
# --------------------------------------------------

spark.stop()