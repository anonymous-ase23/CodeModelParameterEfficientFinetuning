# ----- MAM adapter -----
attn_mode="none"
attn_option="none"
attn_composition="add"
attn_bn=16  # attn bottleneck dim

ffn_mode="adapter"
ffn_option="parallel"
ffn_adapter_layernorm_option="none"
ffn_adapter_init_option="lora"
ffn_adapter_scalar="2"
ffn_bn=28 # ffn bottleneck dim

# ----- lora -----
# attn_mode="lora"
# attn_option="none"
# attn_composition="add"
# attn_bn=16

# set ffn_mode to be 'lora' to use
# lora at ffn as well

# ffn_mode="lora"
# ffn_option="none"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="bert"
# ffn_adapter_scalar="1"
# ffn_bn=16

lora_alpha=32
lora_dropout=0.05
lora_init="lora"

bsz=40
lr=2e-4
max_grad_norm=1
weight_decay=0.1
warmup_updates=0
warmup_ratio=0.06
max_steps=-1
num_train_epochs=10
max_tokens_per_batch=0
max_seq_length=512

lr_scheduler_type="polynomial"
#metric=bleu
unfreeze='ef_'
max_eval_samples=1600
logging_steps=1000

eval_strategy="epoch"
save_steps=5000


model_name='Salesforce/codet5-base'
seed=36
exp_name=${model_name}.am_${attn_mode}.ao_${attn_option}.fm_${ffn_mode}
exp_name+=.fo_${ffn_option}.abn${preseqlen}.fbn${ffn_bn_len}.ac_${attn_composition}
exp_name+=.fl_${ffn_adapter_layernorm_option}.finit_${ffn_adapter_init_option}
exp_name+=.fs_${ffn_adapter_scalar}.unfrz_${unfreeze}.ne${num_train_epochs}.seed_${seed}
SAVE=checkpoints/${exp_name}

mkdir -p ./${SAVE}/cache_data

CUDA_VISIBLE_DEVICES=0 python run_gen.py    \
    --do_test  \
    --seed ${seed}  \
    --save_last_checkpoints \
    --always_save_model   \
    --task summarize \
    --sub_task java \
    --model_type codet5 \
    --data_num -1    \
    --num_train_epochs ${num_train_epochs} \
    --warmup_steps 1000 \
    --learning_rate ${lr}\
    --patience 2   \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --lora_init ${lora_init} \
    --attn_mode ${attn_mode} \
    --attn_option ${attn_option} \
    --attn_composition ${attn_composition} \
    --ffn_mode ${ffn_mode} \
    --ffn_option ${ffn_option} \
    --ffn_adapter_layernorm_option ${ffn_adapter_layernorm_option} \
    --ffn_adapter_scalar ${ffn_adapter_scalar} \
    --ffn_adapter_init_option ${ffn_adapter_init_option} \
    --mid_dim 800 \
    --attn_bn ${attn_bn} \
    --ffn_bn ${ffn_bn} \
    --unfreeze_params ${unfreeze} \
    --tokenizer_name=${model_name}\
    --tokenizer_path=${model_name}   \
    --model_name_or_path=${model_name} \
    --output_dir ${SAVE}/  \
    --summary_dir tensorboard   \
    --data_dir dataset  \
    --cache_path ${SAVE}/cache_data \
    --res_dir ${SAVE}/prediction \
    --res_fn ${SAVE}/summarize_base.txt   \
    --train_batch_size 28 \
    --eval_batch_size 32 \
    --max_source_length 256 \
    --max_target_length 128   \
    2>&1 | tee ${SAVE}/log.txt
