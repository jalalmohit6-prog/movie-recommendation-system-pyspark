from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import IDF
from pyspark.ml.feature import Normalizer
from pyspark.ml.recommendation import ALS


# ==================================================
# 1. CREATE SPARK SESSION
# ==================================================

spark = SparkSession.builder \
    .appName("Hybrid Movie Recommendation System") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# ==================================================
# 2. HDFS PATHS
# ==================================================

movies_path = \
    "hdfs://localhost:9000/movie_recommendation/movies/movies.csv"

ratings_path = \
    "hdfs://localhost:9000/movie_recommendation/ratings/ratings.csv"


# ==================================================
# 3. READ DATA
# ==================================================

movies = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv(movies_path)

ratings = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv(ratings_path)


# ==================================================
# 4. CLEAN DATA
# ==================================================

movies = movies.dropDuplicates(
    ["movieId"]
)

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

movies = movies.fillna({
    "genres": "",
    "director": "",
    "actors": ""
})


print("\n========== DATA ==========")

print("Movies:", movies.count())
print("Ratings:", ratings.count())


# ==================================================
# PART A
# COLLABORATIVE FILTERING
# ==================================================

print("\n========== COLLABORATIVE FILTERING ==========")


train, test = ratings.randomSplit(
    [0.8, 0.2],
    seed=42
)


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


# Generate top 10 recommendations for every user

cf_recommendations = model.recommendForAllUsers(10)


# Convert array into rows

cf_recommendations = cf_recommendations \
    .withColumn(
        "recommendation",
        F.explode("recommendations")
    )


cf_recommendations = cf_recommendations.select(
    "userId",

    F.col(
        "recommendation.movieId"
    ).alias("movieId"),

    F.col(
        "recommendation.rating"
    ).alias("cf_score")
)


print("\nCollaborative recommendations:")

cf_recommendations.show(
    20,
    truncate=False
)


# ==================================================
# PART B
# CONTENT-BASED FILTERING
# ==================================================

print("\n========== CONTENT BASED FILTERING ==========")


# Combine movie information

movies = movies.withColumn(
    "features",
    F.concat_ws(
        " ",
        F.col("genres"),
        F.col("director"),
        F.col("actors")
    )
)


# Tokenizer

tokenizer = Tokenizer(
    inputCol="features",
    outputCol="words"
)

words_data = tokenizer.transform(
    movies
)


# TF

hashing_tf = HashingTF(
    inputCol="words",
    outputCol="rawFeatures",
    numFeatures=1000
)

tf_data = hashing_tf.transform(
    words_data
)


# IDF

idf = IDF(
    inputCol="rawFeatures",
    outputCol="tfidf"
)

idf_model = idf.fit(
    tf_data
)

tfidf_data = idf_model.transform(
    tf_data
)


# Normalize

normalizer = Normalizer(
    inputCol="tfidf",
    outputCol="content_vector",
    p=2.0
)

content_data = normalizer.transform(
    tfidf_data
)


# ==================================================
# CONTENT SIMILARITY
# ==================================================

# We use Toy Story (movieId = 1)
# as the example selected movie.

selected_movie_id = 1


selected_movie = content_data.filter(
    F.col("movieId") == selected_movie_id
).first()


selected_vector = selected_movie[
    "content_vector"
]


print(
    "\nSelected Movie:",
    selected_movie["title"]
)


# Cosine similarity function

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
    lambda v:
        cosine_similarity(
            selected_vector,
            v
        ),
    DoubleType()
)


# Calculate similarity

content_scores = content_data \
    .withColumn(
        "content_score",
        similarity_udf(
            F.col("content_vector")
        )
    ) \
    .select(
        "movieId",
        "content_score"
    )


# ==================================================
# PART C
# NORMALIZE ALS SCORE
# ==================================================

max_cf = cf_recommendations.select(
    F.max("cf_score")
).first()[0]

min_cf = cf_recommendations.select(
    F.min("cf_score")
).first()[0]


if max_cf != min_cf:

    cf_recommendations = \
        cf_recommendations.withColumn(
            "cf_normalized",
            (
                F.col("cf_score") - min_cf
            ) /
            (
                max_cf - min_cf
            )
        )

else:

    cf_recommendations = \
        cf_recommendations.withColumn(
            "cf_normalized",
            F.lit(0.5)
        )


# ==================================================
# PART D
# COMBINE BOTH MODELS
# ==================================================

hybrid = cf_recommendations.join(
    content_scores,
    "movieId",
    "left"
)


hybrid = hybrid.fillna({
    "content_score": 0.0
})


# ==================================================
# HYBRID SCORE
# ==================================================

# 70% Collaborative Filtering
# 30% Content Based Filtering

hybrid = hybrid.withColumn(
    "hybrid_score",
    (
        0.7 * F.col("cf_normalized")
        +
        0.3 * F.col("content_score")
    )
)


# ==================================================
# ADD MOVIE INFORMATION
# ==================================================

hybrid = hybrid.join(
    movies.select(
        "movieId",
        "title",
        "genres"
    ),
    "movieId",
    "left"
)


# ==================================================
# FINAL RECOMMENDATIONS
# ==================================================

print(
    "\n========== HYBRID RECOMMENDATIONS =========="
)


hybrid.select(
    "userId",
    "movieId",
    "title",
    "genres",
    "cf_score",
    "content_score",
    "hybrid_score"
).orderBy(
    "userId",
    F.desc("hybrid_score")
).show(
    50,
    truncate=False
)


# ==================================================
# SAVE HYBRID RESULT TO HDFS
# ==================================================

output_path = \
    "hdfs://localhost:9000/movie_recommendation/hybrid_output"


hybrid.select(
    "userId",
    "movieId",
    "title",
    "genres",
    "cf_score",
    "content_score",
    "hybrid_score"
).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)


print(
    "\nHybrid recommendations saved to HDFS:"
)

print(output_path)


# ==================================================
# STOP SPARK
# ==================================================

spark.stop()