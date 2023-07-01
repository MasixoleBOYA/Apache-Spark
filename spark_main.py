import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("real_project").getOrCreate()

doc = spark.read.csv("<working directory>/income.csv",header = True, inferSchema= True)
doc.printSchema()

doc = doc.na.drop()

label_indexer = StringIndexer(inputCol="income_class", outputCol="Labels").fit(doc)
doc = label_indexer.transform(doc)

categorical_features_temp = [i[0] for i in doc.dtypes if i[1] == "string" and i[0]!= "income_class"]#list of all string-type features
print(f"Categorical Features: {categorical_features_temp}")

categorical_features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'citizenship']#copy and paste the "categorical features" list printed above
remaining_features =['age', 'weight', 'education_years', 'capital_gain', 'capital_loss', 'hours_per_week']#non-categorical features

feature_indexer = [StringIndexer(inputCol=j, outputCol=f"{j}_index").fit(doc) for j in categorical_features]

assembler = VectorAssembler(inputCols=[f"{feature}_index" for feature in categorical_features] + remaining_features,outputCol='features')

scaler = MinMaxScaler(inputCol='features', outputCol='scaled_features')

pipeline = Pipeline(stages=feature_indexer + [assembler, scaler])
new_doc = pipeline.fit(doc).transform(doc)

train, test = new_doc.randomSplit([0.70, 0.30])
print(f"Train Size: {train.count()}")
print(f"Test Size: {test.count()}")

rf = RandomForestClassifier(featuresCol="scaled_features",labelCol="Labels",numTrees=100, maxDepth=10)
rfModel = rf.fit(train)
predictions = rfModel.transform(test)

evaluator = MulticlassClassificationEvaluator(labelCol="Labels",predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print(f"\nAccuracy: {round(accuracy*100,2)}%")
print(f"Error: {round(1.0 - accuracy,4)}")

from pyspark.sql.types import FloatType

preds = predictions.select(["prediction", "Labels"]).withColumn("Labels", F.col("Labels").cast(FloatType()))
preds = preds.select(["prediction", "Labels"])

confusion_matrix = preds.groupBy("Labels").pivot("prediction").count().na.fill(0).orderBy("Labels")
print("\nConfusion Matrix")
confusion_matrix.show()