import numpy as np
import pandas as pd
import time
import logging
import os
import shutil
import uuid
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col, mean, stddev, when
from pyspark.ml.classification import DecisionTreeClassifier # Import Decision Tree Classifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)

def dataset_prep(dataset_path, columns_no, spark):
    try:
        data_schema = StructType([StructField(f"feature_{i+1}", DoubleType(), True) for i in range(columns_no - 1)] +
                                 [StructField("label", StringType(), True)])
        # Create a schema for the dataset with the specified number of columns
        dt_raw = spark.read.format("csv").option("delimiter", ",").option("header", "false").schema(data_schema).load(dataset_path)
        label_indexer = StringIndexer(inputCol="label", outputCol="label_indexed")
        dataset_indexed = label_indexer.fit(dt_raw).transform(dt_raw)
        print("Data Prepared complete.")
        return dataset_indexed
    except Exception as e:
        print(f"Error in dataset_prep: {str(e)}")
        raise

def improve_data_quality(dataset_indexed, columns_no):
    feature_columns = [f"feature_{i+1}" for i in range(columns_no - 1)]
    processed_dataset = dataset_indexed
    print("Handling missing values...")
    for feature_col in feature_columns:
        mean_value = processed_dataset.select(mean(col(feature_col))).collect()[0][0]
        if mean_value is not None:
            processed_dataset = processed_dataset.withColumn(feature_col, when(col(feature_col).isNull(), mean_value).otherwise(col(feature_col)))
    print("Removing outliers...")
    for feature_col in feature_columns:
        stats = processed_dataset.select(mean(col(feature_col)).alias("mean"), stddev(col(feature_col)).alias("stddev")).collect()[0]
        mean_value, stddev_value = stats["mean"], stats["stddev"]
        if stddev_value is not None and stddev_value != 0:
            lower_bound = mean_value - 3 * stddev_value
            upper_bound = mean_value + 3 * stddev_value
            processed_dataset = processed_dataset.filter((col(feature_col) >= lower_bound) & (col(feature_col) <= upper_bound))
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    dataset_with_features = assembler.transform(processed_dataset)
    print("Data quality improvement complete.")
    return dataset_with_features

def decision_tree(train_data, test_data, seed):
    # PARAMETERS:-----------------------------------
    # (maxDepth=5), This is an important hyperparameter. It limits how deep the decision tree can grow
    # (impurity="gini"), This is the criterion used to measure the quality of a split. "gini" is a common choice for classification tasks.

    # INITIALIZATION:--------------------
    # Initialize the Decision Tree Classifier with label and feature columns
    #spark = SparkSession.builder.appName("dt_session").getOrCreate()
    dt_classifier = DecisionTreeClassifier(labelCol="label_indexed", featuresCol="features", seed=seed) 

    # FITTING:----------------------------
    dt_model = dt_classifier.fit(train_data) # Fit the model on the training data

    # PREDICTION:-------------------------
    test_pred = dt_model.transform(test_data) # Make predictions on the test data
    train_pred = dt_model.transform(train_data)  # Make predictions on the training data

    # EVALUATION:-------------------------
    # Evaluate the model's accuracy on the test data
    evaluator = MulticlassClassificationEvaluator(labelCol="label_indexed", 
                                                  predictionCol="prediction", 
                                                  metricName="f1")
    # Evaluate the predictions 
    test_acc = evaluator.evaluate(test_pred)
    train_acc = evaluator.evaluate(train_pred)


    # Show predictions:
    #predictions.select("features", "label_indexed", "prediction").show(5, truncate=False)

    return test_acc, test_pred, train_acc, train_pred  # Return the accuracy and predictions for further analysis

def summary(accuracies):
    # Print a summary includes Min, MAX, Mean, and Standard Deviation
    #accuracies = np.array(accuracies)
    #print("Summary of Accuracies:")
    print(f"Min: {np.min(accuracies):.4f}")
    print(f"Max: {np.max(accuracies):.4f}")
    print(f"Mean: {np.mean(accuracies):.4f}")
    print(f"SD: {np.std(accuracies):.4f}") 

def save_to_csv(test_acc_li, train_acc_li, seeds, spark_session, filename="dt_results.csv"):
    try:
        header = ["Run", "Seed", "Train", "Test"]

        # Create a pandas DataFrame
        results_df = pd.DataFrame({
            "Run": range(1, len(seeds) + 1),
            "Seed": seeds,
            "Train Accuracy": train_acc_li,
            "Test Accuracy": test_acc_li
        }, columns=header)

        # Convert pandas DataFrame to Spark DataFrame
        results_spark_df = spark_session.createDataFrame(results_df)

        # Temporary directory for saving the CSV
        temp_dir = f"temp_output_{uuid.uuid4().hex}"

        # Save the Spark DataFrame to a temporary directory
        results_spark_df.coalesce(1).write.mode("overwrite").csv(temp_dir, header=True)

        # Rename the output file to the specified filename
        for file in os.listdir(temp_dir):
            if file.startswith("part-") and file.endswith(".csv"):
                shutil.move(os.path.join(temp_dir, file), filename)

        # Remove the temporary directory
        shutil.rmtree(temp_dir)

        print(f"Results successfully saved to {filename}")
    except Exception as e:
        print(f"Error writing results to CSV: {str(e)}")

def print_results_table(test_acc_li, train_acc_li, seeds):
    """
    Create a DataFrame table from test_acc_li, train_acc_li, and seeds, 
    and print it as a formatted table in the terminal.
    """
    try:
        # Create a pandas DataFrame
        results_df = pd.DataFrame({
            "Run": range(1, len(seeds) + 1),
            "Seed": seeds,
            "Train Accuracy": train_acc_li,
            "Test Accuracy": test_acc_li
        })

        # Print the DataFrame as a formatted table
        print("\n------- Results Table ----")
        print(results_df.to_string(index=False, float_format="{:.4f}".format))
        print("----------------------------------------------------")
    except Exception as e:
        print(f"Error printing results table: {str(e)}")

def main():
    # -------------------------- PARAMETER CONFIG --------------------------------
    data_path = "kdd.data"
    columns_no = 42  # Number of columns in the dataset 
    seeds = [42, 123, 456, 789, 795, 810, 850, 880, 950, 1000]
    train_acc_li = []  # List to store training accuracies
    test_acc_li = []  # List to store test accuracies
    run_no = 1

    # -------------------------- SPARK SESSION INITIALIZATION --------------------
    spark_session = SparkSession.builder.appName("spark_session").getOrCreate()
    spark_session.sparkContext.setLogLevel("ERROR")

    # -------------------------- DATA PREPARATION -------------------------------
    print("\n------------------- DATA PREPARATION -----------------------")
    print("\n----- STEP 1: Initial Data Preparation ------")
    dataset_indexed = dataset_prep(data_path, columns_no, spark_session)

    print("\n----- STEP 2: Data Cleaning ------")
    dataset_cleaned = improve_data_quality(dataset_indexed, columns_no)

    # Start timing the loop
    start_time = time.time()
    print("\n--------------- RUNNING DECISION TREE CLASSIFIER ---------------\n")
    print("----- Running model with different seeds ------\n")
    for seed in seeds:  
        # Spliting the dataset into training and test sets
        (training_data, test_data) = dataset_cleaned.randomSplit([0.7, 0.3], seed=seed)  # Randomly split the dataset into training and test sets

        # ----------------------- CLASSIFIER ----------------------------------
        test_acc, test_pred, train_acc, train_pred = decision_tree(training_data, test_data, seed)
        test_acc_li.append(round(test_acc, 4))
        train_acc_li.append(round(train_acc, 4))

        print(f"Run {run_no}, with seed({seed}) :")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}\n")

        run_no += 1

    # End timing the loop
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total running time for all loops: {total_time:.2f} seconds")

    # Save and show the results
    #save_to_csv(test_acc_li, train_acc_li, seeds[0:2], spark_session, filename="dt_results.csv")

    # -------------------------- PRINTING  ----------------------------------
    
    # Print Summaraization of results
    print("\n-------------------- SUMMARY OUTPUTS --------------------------\n")
    print("\n---- TRAIN SET SUMMARY -------")
    summary(train_acc_li)

    print("\n\n---- TEST SET SUMMARY -------")
    summary(test_acc_li)

    spark_session.stop()
    
    # Print the results table
    print_results_table(test_acc_li, train_acc_li, seeds)


if __name__ == "__main__":
    main()
    #spark.stop()  # Stop the Spark session if needed
