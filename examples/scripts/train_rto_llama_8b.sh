SFT=OpenRLHF/Llama-3-8b-sft-mixture
DPO=RTO-RL/Llama3-8B-DPO
RM=RTO-RL/Llama3.2-1B-RewardModel
DATASET=(weqweasdas/ultra_train train context_messages)

deepspeed --module openrlhf.cli.train_ppo \
    --pretrain $SFT \
    --critic_pretrain $RM \
    --reward_pretrain $RM --normalize_reward \
    --dpo_pretrain $DPO \
    --prompt_data ${DATASET[0]} --prompt_split ${DATASET[1]} --input_key ${DATASET[2]} \
    --prompt_max_len 1024 --generate_max_len 1024 --apply_chat_template \
    --rollout_batch_size 1024 --train_batch_size 128 --micro_rollout_batch_size 4 --micro_train_batch_size 1 \
    --init_kl_coef 0.01 --actor_learning_rate 5e-7 --critic_learning_rate 9e-6 \
    --dpo_reward_scale 0.05 --dpo_reward_clip 0.05 \
    --zero_stage 2 --adam_offload --flash_attn --gradient_checkpointing --bf16
