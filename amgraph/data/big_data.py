from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
import torch
from ogb.nodeproppred import PygNodePropPredDataset


spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

hdfs_root = "/user/zhangxiaochuan-jk/papers/umtp/papers100M"

edges_orc = f"{hdfs_root}/edges.orc"
labels_orc = f"{hdfs_root}/labels.orc"  # type in ['train', 'valid', 'test']
features_orc = f"{hdfs_root}/features.orc"  # has_label in [0, 1]

mlp_model_path = f"{hdfs_root}/cv_model"


def pt2orc():

    d = PygNodePropPredDataset(name='ogbn-papers100M', root="/home/j-zhangxiaochuan-jk/data/papers100M-bin")

    # edges
    edge_num = d.data.edge_index.size(1)
    step = edge_num // 10
    edge_index = d.data.edge_index.transpose(1, 0)
    for i in range(11):
        spark.createDataFrame(sc.parallelize(edge_index[i*step:(i+1)*step].numpy().tolist(), 100), schema="src: long, dst: long", verifySchema=False).write.mode("overwrite").orc(f"{edges_orc}/slice={i}")

    # labels
    y = d.data.y
    slices = d.get_idx_split()
    types = ['train', 'valid', 'test']
    for t in types:
        labels = torch.cat((slices[t], y[slices[t]]), 1).numpy()
        spark.createDataFrame(sc.parallelize(labels), schema="id: long, label: int").write.mode("overwrite").orc(f"{hdfs_root}/labels.orc/type={t}")

    # remove features of labeled nodes
    mask = torch.zeros(d.data.num_nodes, dtype=torch.long)
    mask[torch.cat((slices['train'], slices['valid'], slices['test']))] = 1
    idx = torch.arange(d.data.num_nodes, dtype=torch.long)
    features = torch.cat((idx[mask], d.data.x[mask]), 1).numpy()
    spark.createDataFrame(sc.parallelize(features), schema="id: long, features: array<float>").write.mode("overwrite").orc(f"{hdfs_root}/features.orc/has_label=1")
    features = torch.cat((idx[~mask], d.data.x[~mask]), 1).numpy()
    spark.createDataFrame(sc.parallelize(features), schema="id: long, features: array<float>").write.mode("overwrite").orc(f"{hdfs_root}/features.orc/has_label=0")

    # infos
    num_labels = sum(slices[t].size()[0] for t in types)
    print(num_labels)


def mlp(df: DataFrame, seed: int = 0) -> float:
    classifier = MultilayerPerceptronClassifier(layers=[128, 128, 128, 172])
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    cv = CrossValidator(estimator=classifier, evaluator=evaluator, numFolds=5, seed=seed)
    model = cv.fit(df)
    model.write().save(mlp_model_path)
    return model.avgMetrics[0]


if __name__ == "__main__":
    pt2orc()

