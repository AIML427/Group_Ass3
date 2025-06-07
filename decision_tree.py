import numpy as np
import pandas as pd
import logging
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col
from pyspark.ml.classification import DecisionTreeClassifier # Import Decision Tree Classifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

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
        #spark = SparkSession.builder.appName("Dataset_Preparation").getOrCreate()

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

        # FEATURE ASSEMBLER:---------------------------------
        # Create a vector assembler to combine the feature columns into a single vector column
        feature_columns = [f"feature_{i+1}" for i in range(columns_no - 1)]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        
        dataset = assembler.transform(dataset_indexed) # Transform the DataFrame to include the 'features' vector column
        
        # PRINTING --------------------------------------
        # Show the first few rows of the prepared dataset
        #dataset.show(truncate=False)
        #dataset.printSchema()
        #dataset.select("features","label_indexed").show(5, truncate=False)
        #print("\nFirst 5 rows of Original DataFrame:")
        #dt_prep.show(5)
        #dataset.select(col("features").getItem(0).alias("first_feature_value")).show(2)

        #spark.stop()  # Stop the Spark session after use
        return dataset  # Return the prepared dataset for further processing
    
    except Exception as e:
        print(f"Error initializing Spark session: {str(e)}")
        raise

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
                                                  metricName="accuracy")
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
    seeds = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627]  
    train_acc_li = []  # List to store training accuracies
    test_acc_li = []  # List to store test accuracies

    spark_session = SparkSession.builder.appName("spark_session").getOrCreate()
    spark_session.sparkContext.setLogLevel("ERROR")

    # -------------------------- DATA PREPARATION -------------------------------
    # Prepare the dataset by reading and transforming it
    dataset = dataset_prep(data_path, columns_no, spark_session)
    # Spliting the dataset into training and test sets
    (training_data, test_data) = dataset.randomSplit([0.7, 0.3], seed=42)  # Randomly split the dataset into training and test sets
    
    #spark_prep.stop()  # Stop the Spark session after preparation

    run_no = 1
    print("\n------- Running Decision Tree Classifier with different seeds...\n")
    for seed in seeds:  
        # ----------------------- Decision Tree ----------------------------------
        test_acc, test_pred, train_acc, train_pred = decision_tree(training_data, test_data, seed)
        test_acc_li.append(test_acc)
        train_acc_li.append(train_acc)
        print(f"Run {run_no}, with seed({seed}) :")
        print(f"Train Accuracy: {test_acc}")
        print(f"Test Accuracy: {train_acc}\n")

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
