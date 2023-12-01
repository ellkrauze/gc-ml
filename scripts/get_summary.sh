#!/bin/bash

JDK="${1:-"`realpath jdk-11.0.20.8`"}"
GC_VIEWER="${2:-"gcviewer-1.36.jar"}"
BM="${3:-"avrora"}"
INPUT_DIR="${4:-"${BM}_logs"}"

output_dir="summaries_${BM}"
ext=".log"

mkdir ${output_dir}
# cd ${output_dir}
for log_file in `find ${INPUT_DIR} -type f`;
do
    log=`basename ${log_file} ${ext}`
    ${JDK}/bin/java -cp ${GC_VIEWER} com.tagtraum.perf.gcviewer.GCViewer ${log_file} ${output_dir}/summary_${log}.csv -t SUMMARY
done
