# Movie Recommendation System using PySpark

A hybrid movie recommendation system built with **Python and PySpark**. The project combines user ratings and movie information to recommend movies using three approaches: **Collaborative Filtering, Content-Based Filtering, and Hybrid Recommendation**.

The collaborative filtering part uses **ALS (Alternating Least Squares)**, while the content-based approach uses movie metadata with **TF-IDF and Cosine Similarity**. The final recommendation combines both approaches to make the recommendations more useful.

## What This Project Does

The system works with two types of information:

* User ratings
* Movie details such as genres, directors, and actors

Based on this data, it can generate recommendations using:

1. **Collaborative Filtering** – recommends movies based on user rating patterns.
2. **Content-Based Filtering** – recommends movies similar to movies based on their metadata.
3. **Hybrid Recommendation** – combines both methods into a single recommendation score.

## Technologies Used

* Python
* PySpark
* Apache Spark MLlib
* HDFS
* TF-IDF
* Cosine Similarity
* ALS
* Matplotlib
* CSV

## Project Structure

```text
movies_recommendation/
│
├── content_filter.py
├── evaluation.py
├── hybrid_recommendation.py
├── movie_recommendation.py
├── visualize.py
│
├── data/
│   ├── movies.csv
│   └── ratings.csv
│
└── output/
```

## Dataset

The project uses two CSV files.

### movies.csv

Contains movie information such as:

* Movie ID
* Movie title
* Genres
* Director
* Actors

### ratings.csv

Contains user rating information:

* User ID
* Movie ID
* Rating

The ratings are used by the ALS model to learn user preferences.

## Data Preprocessing

Before training the recommendation model, the rating data is prepared by:

* Removing duplicate user-movie ratings
* Removing rows with missing user IDs, movie IDs, or ratings
* Splitting the data into training and testing sets

An **80/20 train-test split** is used for model evaluation.

## Collaborative Filtering with ALS

For collaborative filtering, the project uses **ALS (Alternating Least Squares)** from PySpark MLlib.

The model uses the following settings:

```python
rank=10
maxIter=10
regParam=0.1
nonnegative=True
implicitPrefs=False
coldStartStrategy="drop"
```

The trained model is used to generate the **top 5 recommendations for users**.

## Content-Based Recommendation

The content-based part uses movie metadata to find movies that are similar to each other.

The project combines:

* Genres
* Director
* Actors

These features are processed using TF-IDF, and movie similarity is calculated using Cosine Similarity.

### Process

```text
Movie Information
       ↓
Combine Genres + Director + Actors
       ↓
Tokenization
       ↓
TF / IDF
       ↓
Feature Vectors
       ↓
Cosine Similarity
       ↓
Similar Movies
```

## Hybrid Recommendation

The hybrid model combines the results from collaborative filtering and content-based filtering.

The current scoring approach gives:

```text
70% → Collaborative Filtering
30% → Content-Based Filtering
```

This allows the system to consider both **user preferences** and **movie characteristics** when generating recommendations.

## Model Evaluation

The project evaluates the recommendation system using:

* **RMSE**
* **Precision**
* **Recall**
* **F1 Score**

For the recommendation evaluation, movies with a rating of **4 or above** are treated as relevant.

### RMSE

RMSE is used to measure the difference between the actual ratings and the ratings predicted by the recommendation model.

### Precision

Shows how many of the recommended movies are relevant.

### Recall

Shows how many relevant movies were successfully recommended.

### F1 Score

Provides a combined measure of Precision and Recall.

## Results Visualization

The `visualize.py` script is used to create graphs for:

* Precision
* Recall
* F1 Score
* RMSE

These graphs provide a quick view of the recommendation model's performance.

> Note: The metric values currently used in `visualize.py` are sample values. They should be replaced with the actual values produced by the evaluation script before using them as final project results.

## HDFS Usage

The project is configured to work with **HDFS** for storing input data and recommendation outputs.

The project uses paths under:

```text
hdfs://localhost:9000/movie_recommendation/
```

The main directories include:

```text
movie_recommendation/
├── movies/
├── ratings/
├── als_output/
├── content_output/
└── hybrid_output/
```

## How to Run

### 1. Install Dependencies

```bash
pip install pyspark matplotlib
```

### 2. Start Hadoop/HDFS

Make sure Hadoop and HDFS are running on your system.

The project expects HDFS at:

```text
hdfs://localhost:9000/
```

### 3. Add the Dataset

Upload the following files to their respective HDFS locations:

```text
movies.csv
ratings.csv
```

### 4. Run Collaborative Filtering

```bash
spark-submit movie_recommendation.py
```

### 5. Run Content-Based Filtering

```bash
spark-submit content_filter.py
```

### 6. Run the Hybrid Model

```bash
spark-submit hybrid_recommendation.py
```

### 7. Evaluate the Model

```bash
spark-submit evaluation.py
```

### 8. Generate Performance Graphs

```bash
python visualize.py
```

## Key Takeaways

This project helped me work with:

* PySpark for distributed data processing
* ALS for collaborative filtering
* TF-IDF for feature extraction
* Cosine Similarity for movie matching
* Hybrid recommendation techniques
* HDFS for data storage
* Model evaluation using RMSE, Precision, Recall and F1 Score

## Future Improvements

Some possible improvements for the project are:

* Tune ALS parameters to improve recommendations
* Improve the movie feature set
* Experiment with different hybrid weights
* Add a web interface for recommendations
* Build an API for serving recommendations
* Deploy the system on a cloud-based Spark environment

## Author

**Mohit Jalal**

**Project:** Movie Recommendation System using PySpark

---

### Project Workflow

```text
User Ratings + Movie Data
          ↓
     Data Cleaning
          ↓
   ┌──────┴──────┐
   ↓             ↓
 ALS Model    Content Model
   ↓             ↓
Collaborative   TF-IDF +
Recommendations Cosine Similarity
   ↓             ↓
   └──────┬──────┘
          ↓
   Hybrid Recommendation
          ↓
   Model Evaluation
          ↓
 RMSE | Precision | Recall | F1
```
