#!/bin/sh

TESTJAVA="${1:-"jdk-11.0.20.8"}"
DACAPO="${2:-"`realpath dacapo-bench.jar`"}"
CALLBACK="${3:-"`realpath callback`"}"
OUTPUT_DIR="${3:-"gc_logs"}"
BM="${4:-"avrora"}"

echo "TEST JAVA: ${TEST_JAVA}"

OPTS="-XX:+UseParallelGC -Xmx32G -Xms32G -XX:SurvivorRatio=130 -XX:TargetSurvivorRatio=66"

mkdir ${OUTPUT_DIR}
# Clean up if logs already exist
rm -rfv $OUTPUT_DIR/gc_${BM}*.log

for ParallelGCThreads in 4 8 12 16 20 24
do
    for MaxTenuringThreshold in 1 4 7 10 13 16
    do
        # -Xlog:gc*=trace:file=gc_${BM}_${ParallelGCThreads}_${MaxTenuringThreshold}.log:tags,time,uptime,level
        export JAVA_OPTS="-cp ${CALLBACK}:${DACAPO} ${OPTS} -XX:ParallelGCThreads=${ParallelGCThreads} -XX:MaxTenuringThreshold=${MaxTenuringThreshold} "
        JAVA_OPTS="$JAVA_OPTS -Xlog:gc*=trace:file=${OUTPUT_DIR}/gc_${BM}_${ParallelGCThreads}_${MaxTenuringThreshold}.log:tags,time,uptime,level"
        ${TESTJAVA}/bin/java ${JAVA_OPTS} -Dvmstat.enable_jfr=no -Dvmstat.csv=yes Harness -v -n 5 -c VMStatCallback $BM
    done
done