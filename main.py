import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col
from pyspark.ml.classification import DecisionTreeClassifier # Import Decision Tree Classifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression


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
    
def decision_tree(train_data, test_data, seed, spark):
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
    predictions = dt_model.transform(test_data) # Make predictions on the test data

    # EVALUATION:-------------------------
    # Evaluate the model's accuracy on the test data
    evaluator = MulticlassClassificationEvaluator(labelCol="label_indexed", 
                                                  predictionCol="prediction", 
                                                  metricName="accuracy")
    # Evaluate the predictions 
    accuracy = evaluator.evaluate(predictions)  

    # Show predictions:
    #predictions.select("features", "label_indexed", "prediction").show(5, truncate=False)

    return accuracy, predictions  # Return the accuracy and predictions for further analysis

def logistic_regression(train_data, test_data, seed, spark):

    # Initialize the Logistic Regression Classifier
    log_reg_classifier = LogisticRegression(labelCol="label_indexed", featuresCol="features")

    # Fit the model on the training data
    log_reg_model = log_reg_classifier.fit(train_data)

    # Make predictions on the test data
    predictions = log_reg_model.transform(test_data)

    # Evaluate the model's accuracy on the test data
    evaluator = MulticlassClassificationEvaluator(labelCol="label_indexed", 
                                                  predictionCol="prediction", 
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    # Show predictions:
    #predictions.select("features", "label_indexed", "prediction").show(5, truncate=False)

    return accuracy, predictions  # Return the accuracy and predictions for further analysis

def main():
    # -------------------------- PARAMETER CONFIG --------------------------------
    data_path = "kdd.data"
    columns_no = 42  # Number of columns in the dataset 
    seeds = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627]  
    dt_accuracies = []  # List to store training accuracies
    log_reg_accuracies = []  # List to store test accuracies

    spark = SparkSession.builder.appName("Dataset_Preparation").getOrCreate()

    # -------------------------- DATA PREPARATION -------------------------------
    # Prepare the dataset by reading and transforming it
    dataset = dataset_prep(data_path, columns_no, spark)
    # Spliting the dataset into training and test sets
    (training_data, test_data) = dataset.randomSplit([0.7, 0.3], seed=42)  # Randomly split the dataset into training and test sets
    

    for seed in seeds:    
        # ----------------------- Decision Tree ----------------------------------
        dt_accuracy, dt_predictions = decision_tree(training_data, test_data, seed, spark)
        dt_accuracies.append(dt_accuracy)
        print(f"Decision Tree Accuracy with seed {seed}: {dt_accuracy}")
        # Initialize Spark session
        #spark_dt = SparkSession.builder.appName("DecisionTree").getOrCreate()
        #spark_dt.stop()

        # ----------------------- Logistic Regression ----------------------------
        log_reg_accuracy, log_reg_predictions = logistic_regression(training_data, test_data, seed, spark)
        log_reg_accuracies.append(log_reg_accuracy)
        print(f"Logistic Regression Accuracy with seed {seed}: {log_reg_accuracy}")
        #spark_log_reg = SparkSession.builder.appName("LogisticRegression").getOrCreate()
        #spark_log_reg.stop()

    
    #dataset_prep(data_path, columns_no) 

if __name__ == "__main__":
    main()
    #spark.stop()  # Stop the Spark session if needed