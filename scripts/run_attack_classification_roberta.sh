export CUDA_VISIBLE_DEVICES=0

python -u attack_classification.py \
--dataset_path data/hc3_all_single_chatgpt \
--target_model roberta \
--target_model_path /data1/mwl/DeepfakeText/TextFooler/adversary/RoBERTa/hc3/all_single/checkpoint-500 \
--max_seq_length 512 \
--batch_size 16 \
--data_size 20 \
--output_dir adv_results/hc3_all_single_chatgpt \
--counter_fitting_embeddings_path /data1/mwl/DeepfakeText/counter-fitting/word_vectors/counter-fitted-vectors.txt \
--counter_fitting_cos_sim_path cos_sim_counter_fitting.npy \
--USE_cache_path /data1/mwl/tf_cache