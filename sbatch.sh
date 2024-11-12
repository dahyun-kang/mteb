#!/usr/bin/env bash

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=learn
#SBATCH --array=0-52
#SBATCH --job-name=mteb
#SBATCH --output=/home/dahyunkang/rsc/mteb/logs/%A_%a.out
#SBATCH --error=/home/dahyunkang/rsc/mteb/logs/%A_%a.err
#SBATCH --mem-per-cpu=7G
#SBATCH --time=120
#SBATCH --signal=USR2@300
#SBATCH --open-mode=append
#SBATCH --qos=lowest
#SBATCH --time=1400
EXPDIR=/home/dahyunkang/rsc/mteb/

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
integer_argument=${SLURM_ARRAY_TASK_ID}
echo "The integer argument for this job is ${integer_argument}"

# Define the array of options
options=(
"AmazonCounterfactualClassification"
"AmazonPolarityClassification"
"AmazonReviewsClassification"
"Banking77Classification"
"EmotionClassification"
"ImdbClassification"
"MTOPDomainClassification"
"MTOPIntentClassification"
"MassiveIntentClassification"
"MassiveScenarioClassification"
"ToxicConversationsClassification"
"TweetSentimentExtractionClassification"
"ArxivClusteringP2P"
"ArxivClusteringS2S"
"BiorxivClusteringP2P"
"BiorxivClusteringS2S"
"MedrxivClusteringP2P"
"MedrxivClusteringS2S"
"RedditClustering"
"RedditClusteringP2P"
"StackExchangeClustering"
"StackExchangeClusteringP2P"
"TwentyNewsgroupsClustering"
"SprintDuplicateQuestions"
"TwitterSemEval2015"
"TwitterURLCorpus"
"AskUbuntuDupQuestions"
"MindSmallReranking"
"SciDocsRR"
"StackOverflowDupQuestions"
"ArguAna"
"CQADupstackAndroidRetrieval"
"CQADupstackEnglishRetrieval"
"CQADupstackGamingRetrieval"
"CQADupstackGisRetrieval"
"CQADupstackMathematicaRetrieval"
"CQADupstackPhysicsRetrieval"
"CQADupstackProgrammersRetrieval"
"CQADupstackStatsRetrieval"
"CQADupstackTexRetrieval"
"CQADupstackUnixRetrieval"
"CQADupstackWebmastersRetrieval"
"CQADupstackWordpressRetrieval"
"ClimateFEVER"
"DBPedia"
"FEVER"
"FiQA2018"
"HotpotQA"
"MSMARCO"
"NFCorpus"
"NQ"
"QuoraRetrieval"
"SCIDOCS"
"SciFact"
"TRECCOVID"
"Touche2020Retrieval.v3"
"BIOSSES"
"SICK-R"
"STS12"
"STS13"
"STS14"
"STS15"
"STS16"
"STS17"
"STS22"
"STSBenchmark"
"SummEval"
)

task=${options[$SLURM_ARRAY_TASK_ID]}

# Execute the script with the current option

PYTHONPATH=.:~/rsc/fairvit_model_train python test.py --model_name sentence-transformers/all-mpnet-base-v2 --revision "" --task "$task"
