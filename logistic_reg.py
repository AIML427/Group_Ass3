import numpy as np
import pandas as pd
import logging
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, ChiSqSelector, PCA
from pyspark.sql.functions import col, mean, stddev, when
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline

logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)

def dataset_prep(dataset_path, columns_no, spark):
    # Reading dataset
    try:
        # COLUMNS INITIALIZATION:-------------------------- 
        # Create a dataset column list.
        data_schema = StructType([StructField(f"feature_{i+1}", DoubleType(), True) for i in range(columns_no - 1)] +
                            [StructField("label", StringType(), True)]) # Label is now StringType

        # SPARK SESSION:-----------------------------------
        # Initialize Spark session
        dt_prep = spark.read.format("csv") \
            .option("delimiter", ",") \
            .option("header", "false") \
            .schema(data_schema) \
            .load(dataset_path)
        
        # TARGET PREPARATION:--------------------------------- 
        # Convert labels ('normal', 'anomaly') to numerical values (0.0, 1.0), using (StringIndexer)
        label_indexer = StringIndexer(inputCol="label", outputCol="label_indexed")
        dataset_indexed = label_indexer.fit(dt_prep).transform(dt_prep) # Fit (StringIndexer) to learn the mapping
        #dataset_indexed.select("label", "label_indexed").show(5)

        return dataset_indexed  # Return the prepared dataset for further processing
    
    except Exception as e:
        print(f"Error initializing Spark session: {str(e)}")
        raise

def improve_data_quality(dataset, columns_no):
    """
    Improve the quality of the dataset by handling missing values and removing outliers,
    while ensuring the output DataFrame retains the original schema for feature columns.
    Normalization is best handled in a Spark ML Pipeline for consistency and to prevent data leakage.
    """
    # Create a list of feature column names (excluding the label column)
    feature_columns = [f"feature_{i+1}" for i in range(columns_no - 1)]

    # Make a copy of the original dataset to perform transformations
    # This isn't strictly necessary if you're chaining operations, but can be helpful for clarity
    processed_dataset = dataset

    # --- HANDLE MISSING VALUES ---
    # Replace missing values in each feature column with its mean
    print("Handling missing values...")
    for feature_col in feature_columns:
        mean_value = processed_dataset.select(mean(col(feature_col))).collect()[0][0]
        if mean_value is not None:
            processed_dataset = processed_dataset.withColumn(
                feature_col,
                when(col(feature_col).isNull(), mean_value).otherwise(col(feature_col))
            )

    # --- REMOVE OUTLIERS ---
    # Remove rows where any feature value is more than 3 standard deviations from the mean
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

    # The dataset now contains the cleaned individual feature columns and the original label/label_indexed columns.
    # The schema for the feature columns should remain as DoubleType, and other columns untouched.
    feature_columns = [f"feature_{i+1}" for i in range(columns_no - 1)]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    dataset = assembler.transform(processed_dataset)
    print("Data quality improvement complete.")
    
    return dataset

def logistic_regression(train_data, test_data, seed, regParam, elasticNetParam):

    # Initialize the Logistic Regression Classifier
    log_reg_classifier = LogisticRegression(labelCol="label_indexed", featuresCol="features", 
                                            regParam=regParam, elasticNetParam=elasticNetParam)

    # Fit the model on the training data
    log_reg_model = log_reg_classifier.fit(train_data)

    # Make predictions on the test data
    test_pred = log_reg_model.transform(test_data)
    train_pred = log_reg_model.transform(train_data)

    # Evaluate the model's accuracy on the test data
    evaluator = MulticlassClassificationEvaluator(labelCol="label_indexed", 
                                                  predictionCol="prediction", 
                                                  metricName="f1")
    test_acc = evaluator.evaluate(test_pred)
    train_acc = evaluator.evaluate(train_pred)

    # Show predictions:
    #predictions.select("features", "label_indexed", "prediction").show(5, truncate=False)

    return test_acc, test_pred, train_acc, train_pred  # Return the accuracy and predictions for further analysis

def find_optimal_params(dataset, seed, enable_feature_selection=False, num_features_to_select=None):
    """
    Perform cross-validation to find the optimal parameters for Logistic Regression.
    Optionally includes feature selection.
    """
    # Spliting the dataset into training and test sets
    (training_data, test_data) = dataset.randomSplit([0.7, 0.3], seed=seed)

    # 1. Initialize the StandardScaler
    # Ensure scaling is applied after feature selection if feature selection is done first.
    # Or as part of a pipeline.
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                            withStd=True, withMean=False)

    # Initialize the Logistic Regression Classifier
    log_reg_classifier = LogisticRegression(labelCol="label_indexed", featuresCol="scaled_features")

    # Build the Pipeline
    pipeline_stages = [scaler]

    if enable_feature_selection:
        if num_features_to_select is None:
            # You need to specify how many features to select if enabling feature selection
            raise ValueError("num_features_to_select must be provided if enable_feature_selection is True")
        # ChiSqSelector for feature selection (suitable for categorical labels and numerical features)
        # Note: Logistic Regression can give feature importances, but ChiSqSelector is a good
        # pre-processing step for feature selection.
        selector = ChiSqSelector(numTopFeatures=num_features_to_select, featuresCol="scaled_features",
                                 outputCol="selected_features", labelCol="label_indexed")
        pipeline_stages.append(selector)
        log_reg_classifier.setFeaturesCol("selected_features") # Update featuresCol for LR

    pipeline_stages.append(log_reg_classifier)
    pipeline = Pipeline(stages=pipeline_stages)

    # Create a parameter grid for hyperparameter tuning
    param_grid = ParamGridBuilder() \
        .addGrid(log_reg_classifier.regParam, [0.0001, 0.01, 0.2, 0.4]) \
        .addGrid(log_reg_classifier.elasticNetParam, [0.0, 0.2, 0.4, 0.6]) \
        .build()

    # Initialize the evaluator
    evaluator = MulticlassClassificationEvaluator(labelCol="label_indexed",
                                                  predictionCol="prediction",
                                                  metricName="f1") # Using f1 as per your primary metric

    # Initialize CrossValidator
    cross_validator = CrossValidator(estimator=pipeline, # Use the pipeline as the estimator
                                     estimatorParamMaps=param_grid,
                                     evaluator=evaluator,
                                     numFolds=5,
                                     seed=seed)

    # Fit the model using cross-validation on the training data
    cv_model = cross_validator.fit(training_data)

    # Extract the best model and its parameters
    best_pipeline_model = cv_model.bestModel
    # The best model from the pipeline will be the last stage (LogisticRegressionModel)
    best_log_reg_model = best_pipeline_model.stages[-1]

    best_reg_param = best_log_reg_model._java_obj.getRegParam()
    best_elastic_net_param = best_log_reg_model._java_obj.getElasticNetParam()

    print(f"Optimal Parameters Found: regParam={best_reg_param}, elasticNetParam={best_elastic_net_param}")

    # Return the best pipeline model to be used for final evaluation on test data
    return best_reg_param, best_elastic_net_param

def summary(accuracies):
    # Print a summary includes Min, MAX, Mean, and Standard Deviation
    #accuracies = np.array(accuracies)
    #print("Summary of Accuracies:")
    print(f"Min: {np.min(accuracies):.4f}")
    print(f"Max: {np.max(accuracies):.4f}")
    print(f"Mean: {np.mean(accuracies):.4f}")
    print(f"SD: {np.std(accuracies):.4f}") 

def save_to_csv(test_acc_li, train_acc_li, seeds, spark_session, filename="results.csv"):
    try:
        results_df = pd.DataFrame({
            "Run": range(1, len(seeds) + 1),
            "Seed": seeds,
            "Train Accuracy": train_acc_li,
            "Test Accuracy": test_acc_li
        })
        results_spark_df = spark_session.createDataFrame(results_df)

        results_spark_df.coalesce(1).write.mode("overwrite").csv("dt_output.csv", header=True)
    except Exception as e:
        print(f"Error writing results to CSV: {str(e)}")

def main():
    # -------------------------- PARAMETER CONFIG --------------------------------
    data_path = "kdd.data"
    columns_no = 42  # Number of columns in the dataset 
    seeds = [42, 123, 456, 789, 795, 810, 850, 880, 950, 1000]  
    train_acc_li = []  # List to store training accuracies
    test_acc_li = []  # List to store test accuracies

    spark_session = SparkSession.builder.appName("spark_session").getOrCreate()
    spark_session.sparkContext.setLogLevel("ERROR")

    # -------------------------- DATA TRANSFORMATION -------------------------------
    print("\n------- STEP 1: Initial Data Preparation ---------")
    # Prepare the dataset by reading and transforming it
    dataset_indexed = dataset_prep(data_path, columns_no, spark_session)
     
    # Improve data quality 
    print("\n------- STEP 2: Data Clearning ---------")
    dataset_cleaned = improve_data_quality(dataset_indexed, columns_no)
    

    # Find optimal parameters for Logistic Regression
    #optimal_regParam, optimal_elasticNetParam = find_optimal_params(dataset_cleaned, seed=42)
    optimal_regParam=0.000010
    optimal_elasticNetParam=0.75
    learning_rate = 0.001  # Learning rate for adjusting regularization parameters
    
    run_no = 1
    print("\n------- Running Logistic Regression Classifier with different seeds ----------\n")
    for seed in seeds:
        # ----------------------- DATA PREPARATION --------------------------------
        # Spliting the dataset into training and test sets
        (training_data, test_data) = dataset_cleaned.randomSplit([0.7, 0.3], seed=seed)  # Randomly split the dataset into training and test sets
        
        # 1. Initialize the StandardScaler
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                                withStd=True, withMean=False) # withMean=True if you want centering
        
        # Scaling the features
        # Fit the scaler ONLY on the training data
        scaler_model = scaler.fit(training_data)

        train_data_scaled = scaler_model.transform(training_data)
        test_data_scaled = scaler_model.transform(test_data)

        # ----------------------- Logistic Regression ----------------------------------
        test_acc, test_pred, train_acc, train_pred = logistic_regression(train_data_scaled, test_data_scaled, seed, optimal_regParam, optimal_elasticNetParam)

        # Adjust parameters based on performance
        if train_acc > test_acc:  # Overfitting detected
            optimal_regParam += learning_rate  # Increase regularization to reduce overfitting
            optimal_elasticNetParam = min(optimal_elasticNetParam + learning_rate, 1.0)  # Move towards L1 regularization
        elif test_acc > train_acc:  # Underfitting detected
            optimal_regParam = max(optimal_regParam - learning_rate, 0.0001)  # Decrease regularization to improve fit
            optimal_elasticNetParam = max(optimal_elasticNetParam - learning_rate, 0.0)  # Move towards L2 regularization
        
        test_acc_li.append(test_acc)
        train_acc_li.append(train_acc)
        print(f"Run {run_no}, with seed({seed}) :")
        print(f"Train Accuracy: {train_acc}")
        print(f"Test Accuracy: {test_acc}\n")

        run_no += 1

    # Save results to CSV
    #save_to_csv(test_acc_li, train_acc_li, seeds, spark_session)
    
    spark_session.stop()

    # Print Summaraization of results
    print("---------------------------------------------")
    print("Training Set Summary:")
    summary(train_acc_li)
    print("---------------------------------------------")
    print("\nTest Set Summary:")
    summary(test_acc_li)


if __name__ == "__main__":
    main()
    #spark.stop()  # Stop the Spark session if needed