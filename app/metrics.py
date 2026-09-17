from prometheus_client import Counter


PREDICTIONS_TOTAL = Counter(
    "ml_predictions_total",
    "Total number of successful ML predictions",
    ["predicted_class"]
)