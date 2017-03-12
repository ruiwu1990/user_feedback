from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# # Load and parse the data file, converting it to a DataFrame.
# data = spark.read.format("libsvm").load("/cse/home/rwu/Desktop/user_feedback/temp/temp_libsvm.txt")

# # Index labels, adding metadata to the label column.
# # Fit on whole dataset to include all labels in index.
# labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# # Automatically identify categorical features, and index them.
# # Set maxCategories so features with > 4 distinct values are treated as continuous.
# featureIndexer =\
#     VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# # Split the data into training and test sets (30% held out for testing)
# (trainingData, testData) = data.randomSplit([0.7, 0.3])

# # Train a RandomForest model.
# rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# # Convert indexed labels back to original labels.
# labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
#                                labels=labelIndexer.labels)

# # Chain indexers and forest in a Pipeline
# pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# # Train model.  This also runs the indexers.
# model = pipeline.fit(trainingData)

# # Make predictions.
# predictions = model.transform(testData)

# # Select example rows to display.
# predictions.select("predictedLabel", "label", "features").show(5)

# # Select (prediction, true label) and compute test error
# evaluator = MulticlassClassificationEvaluator(
#     labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
# accuracy = evaluator.evaluate(predictions)
# print("Test Error = %g" % (1.0 - accuracy))

# rfModel = model.stages[2]
# print(rfModel)  # summary only

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

accuracy = 0

for i in range(30):
  # Load and parse the data file, converting it to a DataFrame.
  data = spark.read.format("libsvm").load("/cse/home/rwu/Desktop/user_feedback/temp/temp_libsvm.txt")
  # Index labels, adding metadata to the label column.
  # Fit on whole dataset to include all labels in index.
  labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
  # Automatically identify categorical features, and index them.
  # Set maxCategories so features with > 4 distinct values are treated as continuous.
  featureIndexer =\
      VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
  # Split the data into training and test sets (30% held out for testing)
  (trainingData, testData) = data.randomSplit([0.7, 0.3])
  # Train a RandomForest model.
  rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)
  # Convert indexed labels back to original labels.
  labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                 labels=labelIndexer.labels)
  # Chain indexers and forest in a Pipeline
  pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])
  # Train model.  This also runs the indexers.
  model = pipeline.fit(trainingData)
  # Make predictions.
  predictions = model.transform(testData)
  # Select example rows to display.
  predictions.select("predictedLabel", "label", "features").show(5)
  # Select (prediction, true label) and compute test error
  evaluator = MulticlassClassificationEvaluator(
      labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
  accuracy = accuracy + evaluator.evaluate(predictions)

print("Test Error = %g" % (1.0 - (accuracy/30)))