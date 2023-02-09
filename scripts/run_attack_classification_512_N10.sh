export CUDA_VISIBLE_DEVICES=1

python -u attack_classification.py \
--dataset_path data/yelp \
--target_model bert \
--target_model_path /data1/mwl/DeepfakeText/TextFooler/adversary/BERT/results/yelp \
--max_seq_length 512 \
--max_perturb_num 10 \
--batch_size 16 \
--counter_fitting_embeddings_path /data1/mwl/DeepfakeText/counter-fitting/word_vectors/counter-fitted-vectors.txt \
--counter_fitting_cos_sim_path cos_sim_counter_fitting.npy \
--USE_cache_path /data1/mwl/tf_cache