from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    when,
    to_timestamp,
    dayofmonth,
    month,
    year,
)
from pyspark.sql.dataframe import DataFrame
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import (
    LogisticRegression,
    NaiveBayes,
    RandomForestClassifier,
    DecisionTreeClassifier,
    LinearSVC,
    MultilayerPerceptronClassifier,
    GBTClassifier
)


from pyspark.ml.pipeline import Pipeline
from pyspark.ml.feature import (
    VectorAssembler,
    StringIndexer,
    StandardScaler,
    OneHotEncoder,
)

from sklearn.compose import ColumnTransformer

# CONSTANTS
VIOLENT_CRIMES = [
    "RAPE 1",
    "RAPE 2",
    "RAPE 3",
    "SODOMY 1",
    "STRANGULATION 1ST",
    "SEXUAL ABUSE",
]
CATEGORICAL_FEATURES = [
    "Perpetrator_Race",
    "Perpetrator_Sex",
    "Perpetrator_Age_Group",
    "Arrest_Borough",
    "Arrest_Day_of_Week",
]
NUMERICAL_FEATURES = ["Latitude", "Longitude", "Arrest_Month", "Is_Weekend"]
TOTAL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def __spark_init() -> SparkSession:
    """Initialize SparkSession"""

    return (
        SparkSession.builder.appName("NYPD Arrest Modeling")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def __init_pipeline_without_negatives() -> VectorAssembler:
    """Initialize pre-processor without negative values for model-support"""
    indexers = [
        StringIndexer(inputCol=col, outputCol=col + "_idx")
        for col in CATEGORICAL_FEATURES
    ]
    encoders = [
        OneHotEncoder(inputCol=col + "_idx", outputCol=col + "_ohe")
        for col in CATEGORICAL_FEATURES
    ]

    num_assembler = VectorAssembler(
        inputCols=["Arrest_Month", "Is_Weekend"], outputCol="numerical_features"
    )
    scaler = StandardScaler(inputCol="numerical_features", outputCol="numerical_scaled")

    final_features = [col + "_ohe" for col in CATEGORICAL_FEATURES] + [
        "numerical_scaled"
    ]
    final_assembler = VectorAssembler(inputCols=final_features, outputCol="features")

    return indexers + encoders + [num_assembler, scaler, final_assembler]


def __init_pipeline() -> VectorAssembler:
    """Initialize pre-processor"""
    indexers = [
        StringIndexer(inputCol=col, outputCol=col + "_idx")
        for col in CATEGORICAL_FEATURES
    ]
    encoders = [
        OneHotEncoder(inputCol=col + "_idx", outputCol=col + "_ohe")
        for col in CATEGORICAL_FEATURES
    ]

    num_assembler = VectorAssembler(
        inputCols=NUMERICAL_FEATURES, outputCol="numerical_features"
    )
    scaler = StandardScaler(inputCol="numerical_features", outputCol="numerical_scaled")

    final_features = [col + "_ohe" for col in CATEGORICAL_FEATURES] + [
        "numerical_scaled"
    ]
    final_assembler = VectorAssembler(inputCols=final_features, outputCol="features")

    return indexers + encoders + [num_assembler, scaler, final_assembler]


def log_prediction_outcome(target: str, predictions):
    logistic_evaluator = MulticlassClassificationEvaluator(
        labelCol=target, predictionCol="prediction", metricName="accuracy"
    )
    accuracy = logistic_evaluator.evaluate(predictions)
    # Precision
    precision = logistic_evaluator.setMetricName("weightedPrecision").evaluate(
        predictions
    )

    # Recall
    recall = logistic_evaluator.setMetricName("weightedRecall").evaluate(predictions)

    # F1 Score
    f1 = logistic_evaluator.setMetricName("f1").evaluate(predictions)

    print("Recall score: ", recall)
    print("F1 score: ", f1)
    print("Precision score: ", precision)
    print("Accuracy score: ", accuracy)


def clean_features(arrest: DataFrame) -> DataFrame:
    """Pre-process dataframe"""
    # Transform to appropriate types and add new columns.
    arrest = arrest.withColumn(
        "Arrest_Date", to_timestamp(col("Arrest_Date"), "MM/dd/yyyy")
    )
    arrest = arrest.withColumn("Arrest_Day", dayofmonth(col("Arrest_Date")))
    arrest = arrest.withColumn("Arrest_Month", month(col("Arrest_Date")))
    arrest = arrest.withColumn("Arrest_Year", year(col("Arrest_Date")))
    arrest = arrest.withColumn(
        "Is_Weekend",
        when(col("Arrest_Day_Of_Week").isin("Saturday", "Sunday"), 1).otherwise(0),
    )
    arrest = arrest.withColumn(
        "Is_Violent_Crime",
        when(col("Offense_Description").isin(VIOLENT_CRIMES), 1).otherwise(0),
    )
    arrest = arrest.withColumn(
        "Is_Felony", when(col("Offense_Category_Code") == "F", 1).otherwise(0)
    )

    # Drop null columns.
    null_features_to_drop = []

    for feature in TOTAL_FEATURES:
        if feature in arrest.columns:
            null_features_to_drop.append(feature)
    arrest = arrest.dropna(subset=null_features_to_drop)

    return arrest


def apply_logistic_regression(target, arrest, pipeline_stages):
    print("First choice: Logistic Regression.")
    lr = LogisticRegression(labelCol=target)
    logistic_pipeline = Pipeline(stages=pipeline_stages + [lr])

    model = logistic_pipeline.fit(arrest)
    predictions = model.transform(arrest)
    log_prediction_outcome(target, predictions)


def apply_naive_bayes(target, arrest, pipeline_stages):
    """Apply Naive Bayes model."""

    print("Second Choice: Naive Bayes")

    nb = NaiveBayes(labelCol=target)
    pipe = Pipeline(stages=pipeline_stages + [nb])
    model = pipe.fit(arrest)
    predictions = model.transform(arrest)

    log_prediction_outcome(target, predictions)


def apply_random_forest(target, arrest, pipeline_stages):
    """Apply random forest model."""
    # Downgrade numpy < 2

    print("Third Choice: Random Forest.")
    rf = RandomForestClassifier(labelCol=target)
    pipe = Pipeline(stages=pipeline_stages + [rf])
    model = pipe.fit(arrest)
    predictions = model.transform(arrest)

    log_prediction_outcome(target, predictions)


def apply_decision_tree(target, arrest, pipeline_stages):
    """Apply Decision Tree Model."""

    print("Fourth Choice: Decision Tree")

    dt = DecisionTreeClassifier(labelCol=target)
    pipe = Pipeline(stages=pipeline_stages + [dt])
    model = pipe.fit(arrest)
    predictions = model.transform(arrest)
    log_prediction_outcome(target, predictions)


def apply_SVM(target: str, arrest: DataFrame, pipeline_stages: ColumnTransformer):
    """Apply spark SVC model"""
    print("Fifth Choice: SVM")
    svm = LinearSVC(labelCol=target)
    pipe = Pipeline(stages=pipeline_stages + [svm])
    model = pipe.fit(arrest)
    predictions = model.transform(arrest)
    log_prediction_outcome(target, predictions)


def apply_mlp(target, arrest, pipeline_stages):
    """Apply Multi-layer Perceptron Classifier"""

    # Mock preprocess to instantiate model
    count = arrest.select(target).distinct().count()
    preprocessing_pipeline = Pipeline(stages=pipeline_stages)
    pre_model = preprocessing_pipeline.fit(arrest)
    transformed_pre_model = pre_model.transform(arrest)

    input_size = transformed_pre_model.select("features").first()["features"].size
    count = transformed_pre_model.select(target).distinct().count()
    layers = [input_size, 8, 6, count]  # Number of nodes in each neural netowrk layer.

    # Apply model
    mlp = MultilayerPerceptronClassifier(
        featuresCol="features", labelCol=target, layers=layers
    )
    pipe = Pipeline(stages=pipeline_stages + [mlp])
    model = pipe.fit(arrest)
    predictions = model.transform(arrest)
    log_prediction_outcome(target, predictions)


def apply_gbt(target, arrest, pipeline_stages):
    print("Seventh choice: Gradient Boosted Trees")
    gbt = GBTClassifier(labelCol=target)
    pipe = Pipeline(stages=pipeline_stages + [gbt])
    model = pipe.fit(arrest)
    predictions = model.transform(arrest)
    log_prediction_outcome(target, predictions)


def algorithms_application(
    target: str, arrest: DataFrame, pipeline_stages: ColumnTransformer
):
    """Apply all learning models."""
    non_negative_pipeline = __init_pipeline_without_negatives()

    apply_logistic_regression(target, arrest, pipeline_stages)
    apply_naive_bayes(target, arrest, non_negative_pipeline)
    apply_random_forest(target, arrest, pipeline_stages)
    apply_decision_tree(target, arrest, pipeline_stages)
    apply_SVM(target, arrest, pipeline_stages)
    apply_mlp(target, arrest, pipeline_stages)
    apply_gbt(target, arrest, pipeline_stages)
    


if __name__ == "__main__":
    spark = __spark_init()
    arrest = spark.read.csv(
        "../dataset/NYPD_Arrest_Data__Year_to_Date_cleaned.csv",
        header=True,
        inferSchema=True,
    )
    arrest = clean_features(arrest)
    pipeline = __init_pipeline()
    print(arrest)

    print("Predicting Violent Crimes!")
    algorithms_application("Is_Violent_Crime", arrest, pipeline)
    print("\nPredicting Felonies!")
    algorithms_application("Is_Felony", arrest, pipeline)
