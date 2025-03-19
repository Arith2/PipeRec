# PipeRec
Preprocessing Pipelines for Recommender Models
1. This project is based on Meta's DLRM model. Please refer to this git repo: https://github.com/facebookresearch/dlrm.
2. This repo only contains the preprocessing pipelins for Parquet files in the CPU side.

Below are detailed explanations of how to run different files.
1. In Meta's dlrm repo, data_utils.py is used to generate the preprocessed file in CPU and data_s_pytorch.py is used to train DLRM model in GPU.
