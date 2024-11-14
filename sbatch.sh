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


# CLIP
# PYTHONPATH=.:~/rsc/fairvit_model_train python test.py --model_name "clip" --task "$task"  --revision ""

# OpenCLIP
# PYTHONPATH=.:~/rsc/fairvit_model_train:~/rsc/fairvit_model_train/app/third_party python test.py --model_name "laion/CLIP-ViT-L-14-laion2B-s32B-b82K" --task "$task"  --revision ""

# DFN
# PYTHONPATH=.:~/rsc/fairvit_model_train:~/rsc/fairvit_model_train/app/third_party python test.py --model_name "DFN2B-CLIP-ViT-L-14" --task "$task"  --revision ""

# SIGLIP
# PYTHONPATH=.:~/rsc/fairvit_model_train python test.py --model_name "google/siglip-so400m-patch14-384" --revision "" --task "$task"

# DINOv2_VLMs
# PYTHONPATH=.:~/rsc/fairvit_model_train python test.py --model_name metaclip --model_root "/checkpoint/dino/cijose/experiments/LiT/cvpr_main_experiments/MetaCLIP_v2_Balanced_090924_Mitigated_Airstore_access_modeITERABLE/vit-mae-large_768d12h12l_224_cls_50000_2000_32768_lr_0.0005_wd_0.0001_1024_dp_0.0_vb_0_vlp_False_beta1_0.9_beta2_0.99_eps_1e-08_fls_False/eval/training_49999" --task "$task"

# MetaCLIPv3
# PYTHONPATH=.:~/rsc/fairvit_ssl:~/rsc/fairvit_ssl/app/third_party/ python test.py --model_name "facebook/metaclip-l14-fullcc2.5b" --revision "" --task  "$task"

# PYTHONPATH=.:~/rsc/fairvit_model_train python test.py --model_name dinov2_vlm --model_root "/checkpoint/dino/cijose/experiments/LiT/cvpr_main_experiments/MetaCLIP_v2_Balanced_090924_Mitigated_Airstore_access_modeITERABLE/ViT-L-14_768d12h12l_224_cls_50000_2000_32768_lr_0.0005_wd_0.2_768_dp_0.0_vb_0_vlp_False_beta1_0.9_beta2_0.98_eps_1e-06_fls_False_quickgelu_bf16" --task "$task" --revision ""
PYTHONPATH=.:~/rsc/fairvit_model_train python test.py --model_name dinov2_vlm --model_root "/checkpoint/dino/dahyunkang/experiments/1101_4phases_3r_ver1_mul1_1500M_ssv2_0vb_dim2048_res224_50k_lr7e-4_sequential3" --task "$task" --revision ""
