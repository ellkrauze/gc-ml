#!/bin/bash

JDK="/home/vsakovskaya/gc-ml/jdk-11.0.20"
# input_dir="/home/vsakovskaya/gc-ml/gc_avrora_logs"
# output_dir="summaries_avrora"
gc_viewer_path="/home/vsakovskaya/gc-ml/gcviewer-1.36.jar"

input_dir="/home/vsakovskaya/gc-ml/gc_kafka_logs/home/bellsoft/vsakovskaya/gc-ml/kafka_logs"
output_dir="summaries_kafka"
ext=".log.0"

cd ${output_dir} 
for log_file in `find ${input_dir} -type f`;
do
    log=`basename ${log_file} ${ext}`
    ${JDK}/bin/java -cp ${gc_viewer_path} com.tagtraum.perf.gcviewer.GCViewer ${log_file} summary_${log}.csv -t SUMMARY
done

# java -cp gcviewer-1.36.jar com.tagtraum.perf.gcviewer.GCViewer gc-avrora.txt summary.csv -t SUMMARY