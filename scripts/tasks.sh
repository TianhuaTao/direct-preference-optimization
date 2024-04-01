##################
## llama ultrafeedback sft (finished)
python -u train.py model=llama2-7b datasets=[ultrafeedback] loss=sft exp_name=ultrafeedback_sft_llama2 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=12800


## llama ultrafeedback dpo
python -u train.py model=llama2-7b datasets=[ultrafeedback] loss=dpo loss.beta=0.1 exp_name=ultrafeedback_dpo_llama2 gradient_accumulation_steps=4 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/home/tianhua3/agpt/agpt-dpo/direct-preference-optimization/.cache/tianhua3/ultrafeedback_sft_llama2_2024-03-31_20-15-30_309905/LATEST/policy.pt eval_every=12800

## llama ultrafeedback dpo (skip sft)
python -u train.py model=llama2-7b datasets=[ultrafeedback] loss=dpo loss.beta=0.1 exp_name=ultrafeedback_dpo-no-sft_llama2 gradient_accumulation_steps=4 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/home/tianhua3/agpt/agpt-dpo/state_dict/Llama-2-7b-hf.pt eval_every=12800


##################
## llama hh sft
python -u train.py model=llama2-7b datasets=[hh] loss=sft exp_name=hh_sft_llama2 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=12800

## llama hh dpo
python -u train.py model=llama2-7b datasets=[hh] loss=dpo loss.beta=0.1 exp_name=hh_dpo_llama2 gradient_accumulation_steps=4 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=??/LATEST/policy.pt eval_every=12800

## llama hh dpo (skip sft)
python -u train.py model=llama2-7b datasets=[hh] loss=dpo loss.beta=0.1 exp_name=hh_dpo-no-sft_llama2 gradient_accumulation_steps=4 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/home/tianhua3/agpt/agpt-dpo/state_dict/Llama-2-7b-hf.pt eval_every=12800

##################
# pythia hh sft (finished)
python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=hh_sft_pythia28 gradient_accumulation_steps=1 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=64000

# pythia hh dpo (finished)
python -u train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 exp_name=hh_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/home/tianhua3/agpt/agpt-dpo/direct-preference-optimization/.cache/tianhua3/hh_sft_pythia28_2024-03-31_20-50-15_031216/LATEST/policy.pt eval_every=64000

# pythia hh dpo (skip sft)
python -u train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 exp_name=hh_dpo-no-sft_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/home/tianhua3/agpt/agpt-dpo/state_dict/Pythia-28-hf.pt eval_every=64000

##################
# pythia ultrafeedback sft
python -u train.py model=pythia28 datasets=[ultrafeedback] loss=sft exp_name=ultrafeedback_sft_pythia28 gradient_accumulation_steps=1 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=64000

# pythia ultrafeedback dpo
python -u train.py model=pythia28 datasets=[ultrafeedback] loss=dpo loss.beta=0.1 exp_name=ultrafeedback_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=??/LATEST/policy.pt eval_every=64000

