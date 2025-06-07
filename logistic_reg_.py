import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev, when
from pyspark.sql.types import DoubleType, StringType, StructField, StructType
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

# --- REVISED FUNCTION 1: dataset_prep ---
def dataset_prep(dataset_path, columns_no, spark):
    """
    Reads the dataset, defines schema, and converts the label to numerical.
    Does NOT assemble features into a vector here, allowing pre-processing of individual columns.
    """
    try:
        data_schema = StructType([StructField(f"feature_{i+1}", DoubleType(), True) for i in range(columns_no - 1)] +
                                 [StructField("label", StringType(), True)])

        dt_raw = spark.read.format("csv") \
            .option("delimiter", ",") \
            .option("header", "false") \
            .schema(data_schema) \
            .load(dataset_path)

        label_indexer = StringIndexer(inputCol="label", outputCol="label_indexed")
        dataset_indexed = label_indexer.fit(dt_raw).transform(dt_raw)
        
        return dataset_indexed # Return DataFrame with individual features and indexed label
    
    except Exception as e:
        print(f"Error in dataset_prep: {str(e)}")
        raise

# --- REVISED FUNCTION 2: improve_data_quality ---
def improve_data_quality(dataset_indexed, columns_no):
    """
    Improve the quality of the dataset by handling missing values and removing outliers
    on individual feature columns. Then, it assembles the cleaned individual features
    into a single 'features' vector column.
    """
    feature_columns = [f"feature_{i+1}" for i in range(columns_no - 1)]
    processed_dataset = dataset_indexed # Start with the indexed dataset

    print("Handling missing values...")
    for feature_col in feature_columns:
        mean_value = processed_dataset.select(mean(col(feature_col))).collect()[0][0]
        if mean_value is not None:
            processed_dataset = processed_dataset.withColumn(
                feature_col,
                when(col(feature_col).isNull(), mean_value).otherwise(col(feature_col))
            )

    print("Removing outliers...")
    for feature_col in feature_columns:
        stats = processed_dataset.select(
            mean(col(feature_col)).alias("mean"),
            stddev(col(feature_col)).alias("stddev")
        ).collect()[0]
        mean_value, stddev_value = stats["mean"], stats["stddev"]

        if stddev_value is not None and stddev_value != 0:
            lower_bound = mean_value - 3 * stddev_value
            upper_bound = mean_value + 3 * stddev_value
            processed_dataset = processed_dataset.filter(
                (col(feature_col) >= lower_bound) & (col(feature_col) <= upper_bound)
            )

    # Assemble the cleaned individual feature columns into a single vector column
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    dataset_with_features = assembler.transform(processed_dataset)
    
    print("Data quality improvement complete and features assembled.")
    return dataset_with_features

# --- REVISED FUNCTION 3: logistic_regression_pipeline_run ---
# This is the function from the previous response, designed to be used in the main loop
def logistic_regression_pipeline_run(train_data, test_data, regParam, elasticNetParam, featuresCol="features", labelCol="label_indexed"):
    """
    Trains and evaluates a Logistic Regression model using specified hyperparameters
    within a Spark ML Pipeline that includes StandardScaler.
    """
    scaler = StandardScaler(inputCol=featuresCol, outputCol="scaled_features",
                            withStd=True, withMean=False)

    log_reg_classifier = LogisticRegression(labelCol=labelCol, featuresCol="scaled_features",
                                            regParam=regParam, elasticNetParam=elasticNetParam)

    pipeline = Pipeline(stages=[scaler, log_reg_classifier])

    #print(f"Training Logistic Regression model with regParam={regParam}, elasticNetParam={elasticNetParam}...")
    model = pipeline.fit(train_data)

    #print("Making predictions on training data...")
    train_pred = model.transform(train_data)

    #print("Making predictions on test data...")
    test_pred = model.transform(test_data)

    evaluator = MulticlassClassificationEvaluator(labelCol=labelCol,
                                                  predictionCol="prediction",
                                                  metricName="f1")

    train_f1 = evaluator.evaluate(train_pred)
    test_f1 = evaluator.evaluate(test_pred)

    #print(f"Train F1-Score: {train_f1:.4f}")
    #print(f"Test F1-Score: {test_f1:.4f}")

    return test_f1, test_pred, train_f1, train_pred

# --- REVISED FUNCTION 4: find_optimal_params ---
def find_optimal_params(training_data_for_cv, seed, enable_feature_selection=False, num_features_to_select=None):
    """
    Perform cross-validation to find the optimal parameters for Logistic Regression.
    Takes a training dataset (already split from main data) as input.
    """
    # The split for train/test for hyperparameter tuning is done internally by CrossValidator
    # on the provided 'training_data_for_cv'.

    scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                            withStd=True, withMean=False)

    log_reg_classifier = LogisticRegression(labelCol="label_indexed", featuresCol="scaled_features")

    pipeline_stages = [scaler]

    if enable_feature_selection:
        if num_features_to_select is None:
            raise ValueError("num_features_to_select must be provided if enable_feature_selection is True")
        selector = ChiSqSelector(numTopFeatures=num_features_to_select, featuresCol="scaled_features",
                                 outputCol="selected_features", labelCol="label_indexed")
        pipeline_stages.append(selector)
        log_reg_classifier.setFeaturesCol("selected_features")

    pipeline_stages.append(log_reg_classifier)
    pipeline = Pipeline(stages=pipeline_stages)

    # Expanded parameter grid for better tuning
    param_grid = ParamGridBuilder() \
        .addGrid(log_reg_classifier.regParam, [1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0, 10.0]) \
        .addGrid(log_reg_classifier.elasticNetParam, [0.0, 0.25, 0.5, 0.75, 1.0]) \
        .build()

    evaluator = MulticlassClassificationEvaluator(labelCol="label_indexed",
                                                  predictionCol="prediction",
                                                  metricName="f1")

    cross_validator = CrossValidator(estimator=pipeline,
                                     estimatorParamMaps=param_grid,
                                     evaluator=evaluator,
                                     numFolds=5, # You can increase this to 10 for more robustness
                                     seed=seed)

    cv_model = cross_validator.fit(training_data_for_cv) # Fit CV on the provided training data

    best_pipeline_model = cv_model.bestModel
    best_log_reg_model = best_pipeline_model.stages[-1]

    best_reg_param = best_log_reg_model._java_obj.getRegParam()
    best_elastic_net_param = best_log_reg_model._java_obj.getElasticNetParam()

    print(f"Optimal Parameters Found: regParam={best_reg_param:.6f}, elasticNetParam={best_elastic_net_param:.2f}")
    return best_reg_param, best_elastic_net_param

# --- HELPER FUNCTIONS (No changes needed) ---
def summary(accuracies):
    print(f"Min: {np.min(accuracies):.4f}")
    print(f"Max: {np.max(accuracies):.4f}")
    print(f"Mean: {np.mean(accuracies):.4f}")
    print(f"SD: {np.std(accuracies):.4f}")

def save_to_csv(test_acc_li, train_acc_li, seeds, spark_session, filename="results.csv"):
    try:
        results_df = pd.DataFrame({
            "Run": range(1, len(seeds) + 1),
            "Seed": seeds,
            "Train F1-Score": train_acc_li, # Corrected header
            "Test F1-Score": test_acc_li    # Corrected header
        })
        results_spark_df = spark_session.createDataFrame(results_df)

        results_spark_df.coalesce(1).write.mode("overwrite").csv(filename, header=True)
    except Exception as e:
        print(f"Error writing results to CSV: {str(e)}")

# --- MAIN FUNCTION (REVISED) ---
def main():
    data_path = "kdd.data"
    columns_no = 42
    seeds = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627]
    train_f1_li = []
    test_f1_li = []

    spark_session = SparkSession.builder.appName("spark_session").getOrCreate()
    spark_session.sparkContext.setLogLevel("ERROR")

    # -------------------------- DATA PREPARATION CHAIN --------------------------
    print("\n-------- STEP 1: Initial Data Preparation ---------")
    dataset_indexed_labels = dataset_prep(data_path, columns_no, spark_session)
    
    print("\n-------- STEP 2: Improving Data Quality -----------")
    # This function now correctly creates the 'features' vector after cleaning
    final_dataset_with_features = improve_data_quality(dataset_indexed_labels, columns_no)

    # -------------------------- HYPERPARAMETER TUNING ---------------------------
    # Perform one main train-test split for the entire experiment
    # The training_data_for_tuning will be used by CrossValidator in find_optimal_params
    (training_data_for_tuning, test_data_final_eval) = final_dataset_with_features.randomSplit([0.7, 0.3], seed=42)

    print("\n-------- STEP 3: Finding Optimal Hyperparameters ---------")
    optimal_regParam, optimal_elasticNetParam = find_optimal_params(
         training_data_for_tuning,
         seed=42, # Use a fixed seed for hyperparameter search reproducibility
         enable_feature_selection=False, # Set to True and define num_features_to_select to enable
         # num_features_to_select=20
    )
    #optimal_regParam = 
    #optimal_elasticNetParam = 
    
    # -------------------------- FINAL MODEL EVALUATION ACROSS SEEDS ----------------
    run_no = 1
    print("\n-------- STEP 4: Running Logistic Regression ----------\n")
    for seed in seeds:
        print(f"--- Run {run_no}, with seed({seed}) ---")
        # Split the ORIGINAL cleaned dataset again for robust evaluation with different seeds
        # This split is for the overall train/test sets for THIS specific run.
        (train_data_for_run, test_data_for_run) = final_dataset_with_features.randomSplit([0.7, 0.3], seed=seed)

        # Use the logistic_regression_pipeline_run which includes StandardScaler
        test_f1, test_pred, train_f1, train_pred = logistic_regression_pipeline_run(
            train_data_for_run,
            test_data_for_run,
            regParam=optimal_regParam,
            elasticNetParam=optimal_elasticNetParam,
            featuresCol="features" # The input column for StandardScaler in the pipeline
        )

        test_f1_li.append(test_f1)
        train_f1_li.append(train_f1)
        
        #print(f"Summary for Run {run_no} (Seed: {seed}):")
        print(f"Train F1-Score: {train_f1:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}\n")

        run_no += 1

    # -------------------------- SUMMARIZE AND SAVE RESULTS --------------------------
    #save_to_csv(test_f1_li, train_f1_li, seeds, spark_session, filename="dt_output.csv")
    
    spark_session.stop()

    print("---------------------------------------------")
    print("Training Set Summary (F1-Score Across Runs):")
    summary(train_f1_li)
    print("---------------------------------------------")
    print("\nTest Set Summary (F1-Score Across Runs):")
    summary(test_f1_li)


if __name__ == "__main__":
    main()