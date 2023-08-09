PWD=$(cd "$(dirname "$0")";pwd) || exit
CUR_NS=$((10#`date '+%s'`*1000+10#`date '+%N'`/1000000))
SPARK_HOME=/usr/hdp/3.1.4.0-315/spark3.3.2
HDFS_SPARK_DIR=hdfs://360jinronglycc/user/zhangxiaochuan-jk/spark


${SPARK_HOME}/bin/spark-submit \
  --master yarn \
  --deploy-mode client \
  --jars "${HDFS_SPARK_DIR}/lib_scala_2.12/*.jar" \
  --conf spark.yarn.maxAppAttempts=1 \
  --conf spark.task.maxFailures=20 \
  --conf spark.sql.shuffle.partitions=2000 \
  --conf spark.default.parallelism=2000 \
  --conf spark.dynamicAllocation.maxExecutors=2000 \
  --conf spark.dynamicAllocation.minExecutors=400 \
  --conf spark.dynamicAllocation.enabled=true \
  --files ${SPARK_HOME}/conf/log4j.properties \
  --conf spark.yarn.dist.archives="${HDFS_SPARK_DIR}/lib/py39.zip#Python" \
  --conf spark.pyspark.python='Python/bin/python' \
  --conf spark.pyspark.driver.python='/home/j-zhangxiaochuan-jk/apps/anaconda3/envs/py39/bin/python' \
  --conf spark.driver.extraJavaOptions="-Dlog4j.configuration=file:log4j.properties" \
  --conf spark.executor.extraJavaOptions="-Dlog4j.configuration=file:log4j.properties" \
  --conf spark.memory.storageFraction=0.4 \
  \
  --queue yushu_offline_data_ai_big_job \
  --name graph_huge_lpa_test_j_zhangxiaochuan_jk_${CUR_NS} \
  --conf spark.yarn.priority=1 \
  --driver-memory 6g \
  --num-executors 1000 \
  --executor-cores 2 \
  --conf spark.task.cpus=1 \
  --executor-memory 6g \
  --conf spark.executor.memoryOverhead=2g \
  "${PWD}"/big_data.py
