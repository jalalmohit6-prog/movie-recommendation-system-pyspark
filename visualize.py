import matplotlib.pyplot as plt


# Replace these values with the
# actual values printed by evaluation.py

rmse = 0.85
precision = 0.40
recall = 0.50
f1 = 0.44


# ----------------------------------------
# Graph 1: Precision, Recall and F1
# ----------------------------------------

metrics = [
    "Precision",
    "Recall",
    "F1 Score"
]

values = [
    precision,
    recall,
    f1
]


plt.figure(figsize=(8, 5))

plt.bar(
    metrics,
    values
)

plt.title(
    "Movie Recommendation Performance"
)

plt.xlabel(
    "Metrics"
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0,
    1
)

plt.show()


# ----------------------------------------
# Graph 2: RMSE
# ----------------------------------------

plt.figure(figsize=(6, 5))

plt.bar(
    ["RMSE"],
    [rmse]
)

plt.title(
    "ALS Model RMSE"
)

plt.xlabel(
    "Metric"
)

plt.ylabel(
    "RMSE"
)

plt.show()