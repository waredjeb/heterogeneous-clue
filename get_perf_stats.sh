#!/bin/bash
CMD=$1
_APP_ARGS=$2
APP_ARGS="${_APP_ARGS:-'--maxEvents 10'}"

FILES="toyDetector_4k toyDetector_5k toyDetector_6k toyDetector_7k toyDetector_8k toyDetector_9k toyDetector_10k"
N_REPETITIONS=10


for FileName in $FILES
do
    >&2 echo "Processing file "
    >&2 echo $FileName
    echo "Processing file $FileName"
    for RNum in $(seq 1 $N_REPETITIONS)
    do
        >&2 echo "Run.." $RNum " with params " "--maxEvents 100 --inputFile data/input/$FileName.csv --numberOfStreams 2"
        $CMD --maxEvents 100 --inputFile data/input/$FileName.csv --numberOfStreams 2 2> /dev/null | tail -1 | cut -d' ' -f 5 
    done
done





# CLUE Heterogeneous {
#     cpu,
#     intel-gpu,
#     nvidia-gpu:
#     {
#         numberOfStreams,
#         numberOfThreads,
#         maxEvents,
#         fileName
#     }
# }
# Params CLI -> Throughput

# CLUE SYCL Standalone {
#     intel-gpu, cpu, nvidia-gpu
#     numberOfRuns
#     fileName
# }
# Params CLI -> n rows * elapsed time

# CLUE Native Standalone {
#     cpu, nvidia-gpu
#     fileName
# }
# Params CLI -> n rows * elapsed time