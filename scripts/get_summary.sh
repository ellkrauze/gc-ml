#!/bin/bash

JDK="${1:-"jdk-11.0.20.8"}"
GC_VIEWER="${2:-"gcviewer-1.36.jar"}"
BM="${3:-"avrora"}"

input_dir="gc-logs"
output_dir="summaries_${BM}"
ext=".log"

cd ${output_dir} 
for log_file in `find ${input_dir} -type f`;
do
    log=`basename ${log_file} ${ext}`
    ${JDK}/bin/java -cp ${GC_VIEWER} com.tagtraum.perf.gcviewer.GCViewer ${log_file} summary_${log}.csv -t SUMMARY
done