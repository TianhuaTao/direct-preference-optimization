# step1 llama base sft
python -u train.py model=llama2-7b datasets=[ultrafeedback] loss=sft exp_name=anthropic_dpo_llama2 gradient_accumulation_steps=8 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=256 n_examples=1024


# step2 llama sft dpo
python -u train.py model=llama2-7b datasets=[hh] loss=dpo loss.beta=0.1 exp_name=anthropic_dpo_llama2 gradient_accumulation_steps=8 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=??? eval_every=256 n_examples=1024
