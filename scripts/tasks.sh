##################
## llama ultrafeedback sft  >> finished (.cache/tianhua3/ultrafeedback_sft_llama2_2024-04-01_01-30-03_581140/LATEST/policy.pt)
python -u train.py model=llama2-7b datasets=[ultrafeedback] loss=sft exp_name=ultrafeedback_sft_llama2 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=12800


## llama ultrafeedback dpo >> finished (.cache/tianhua3/ultrafeedback_dpo_llama2_2024-04-01_02-05-30_347982)
python -u train.py model=llama2-7b datasets=[ultrafeedback] loss=dpo loss.beta=0.1 exp_name=ultrafeedback_dpo_llama2 gradient_accumulation_steps=4 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=.cache/tianhua3/ultrafeedback_sft_llama2_2024-04-01_01-30-03_581140/LATEST/policy.pt eval_every=12800 model.reference_dtype=bfloat16

## llama ultrafeedback dpo (skip sft)  >> finished (.cache/tianhua3/ultrafeedback_dpo-no-sft_llama2_2024-04-01_13-11-00_220708)
python -u train.py model=llama2-7b datasets=[ultrafeedback] loss=dpo loss.beta=0.1 exp_name=ultrafeedback_dpo-no-sft_llama2 gradient_accumulation_steps=8 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=../state_dict/Llama-2-7b-hf.pt eval_every=12800 model.reference_dtype=bfloat16


##################
## llama hh sft >> finished (direct-preference-optimization/.cache/tianhua3/hh_sft_llama2_2024-04-01_08-19-32_314096)
python -u train.py model=llama2-7b datasets=[hh] loss=sft exp_name=hh_sft_llama2 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=12800

## llama hh dpo >> finished (.cache/tianhua3/hh_dpo_llama2_2024-04-01_06-38-44_189131)
python -u train.py model=llama2-7b datasets=[hh] loss=dpo loss.beta=0.1 exp_name=hh_dpo_llama2 gradient_accumulation_steps=4 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=.cache/tianhua3/hh_sft_llama2_2024-04-01_08-19-32_314096/LATEST/policy.pt eval_every=12800 model.reference_dtype=bfloat16

## llama hh dpo (skip sft) >> finished (.cache/tianhua3/hh_dpo-no-sft_llama2_2024-04-01_16-06-42_761861)
python -u train.py model=llama2-7b datasets=[hh] loss=dpo loss.beta=0.1 exp_name=hh_dpo-no-sft_llama2 gradient_accumulation_steps=8 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=../state_dict/Llama-2-7b-hf.pt eval_every=12800 model.reference_dtype=bfloat16

##################
# pythia hh sft >> finished (/home/tianhua3/agpt/agpt-dpo/direct-preference-optimization/.cache/tianhua3/hh_sft_pythia28_2024-03-31_20-50-15_031216)
python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=hh_sft_pythia28 gradient_accumulation_steps=1 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=64000

# pythia hh dpo  >> finished (/home/tianhua3/agpt/agpt-dpo/direct-preference-optimization/.cache/tianhua3/hh_dpo_pythia28_2024-03-31_21-20-06_233890)
python -u train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 exp_name=hh_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/home/tianhua3/agpt/agpt-dpo/direct-preference-optimization/.cache/tianhua3/hh_sft_pythia28_2024-03-31_20-50-15_031216/LATEST/policy.pt eval_every=64000 model.reference_dtype=bfloat16

# pythia hh dpo (skip sft) >> finished (.cache/tianhua3/hh_dpo-no-sft_pythia28_2024-04-03_05-08-08_902418) 
python -u train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 exp_name=hh_dpo-no-sft_pythia28 gradient_accumulation_steps=4 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=../state_dict/pythia-2.8b.pt eval_every=64000 model.reference_dtype=bfloat16

##################
# pythia ultrafeedback sft  >> finished (.cache/tianhua/ultrafeedback_sft_pythia28_2024-04-01_07-31-42_478130/LATEST/policy.pt) Polaris
            #  >> finished (.cache/tianhua3/ultrafeedback_sft_pythia28_2024-04-02_20-40-52_114340)
python -u train.py model=pythia28 datasets=[ultrafeedback] loss=sft exp_name=ultrafeedback_sft_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=64000

# pythia ultrafeedback dpo >> finished (/home/tianhua3/agpt/agpt-dpo/direct-preference-optimization/.cache/tianhua3/ultrafeedback_dpo_pythia28_2024-04-02_20-57-07_074166)
    # >> finished (ultrafeedback_dpo_pythia28_2024-04-02_20-57-07_074166)
python -u train.py model=pythia28 datasets=[ultrafeedback] loss=dpo loss.beta=0.1 exp_name=ultrafeedback_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=.cache/tianhua3/ultrafeedback_sft_pythia28_2024-04-02_20-40-52_114340/LATEST/policy.pt eval_every=64000 model.reference_dtype=bfloat16

# pythia ultrafeedback dpo (skip sft) >> (finished .cache/tianhua/ultrafeedback_dpo-no-sft_pythia28_2024-04-01_10-39-11_934895)
    # >> finished( .cache/tianhua3/ultrafeedback_dpo-no-sft_pythia28_2024-04-02_22-32-13_845112)
python -u train.py model=pythia28 datasets=[ultrafeedback] loss=dpo loss.beta=0.1 exp_name=ultrafeedback_dpo-no-sft_pythia28 gradient_accumulation_steps=4 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=../state_dict/pythia-2.8b.pt eval_every=64000 model.reference_dtype=bfloat16

##################
# OLMo ultrafeedback sft >> finished (.cache/tianhua3/ultrafeedback_sft_olmo_2024-04-02_13-25-27_040353)
python -u train.py model=olmo-7b datasets=[ultrafeedback] loss=sft exp_name=ultrafeedback_sft_olmo gradient_accumulation_steps=8 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=12800x

# OLMo ultrafeedback dpo >> finished (.cache/tianhua3/ultrafeedback_dpo_olmo_2024-04-02_14-06-12_855447)
python -u train.py model=olmo-7b datasets=[ultrafeedback] loss=dpo loss.beta=0.1 exp_name=ultrafeedback_dpo_olmo gradient_accumulation_steps=8 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=.cache/tianhua3/ultrafeedback_sft_olmo_2024-04-02_13-25-27_040353/LATEST/policy.pt eval_every=12800 model.reference_dtype=bfloat16

# OLMo ultrafeedback dpo (skip sft) >> finished (.cache/tianhua3/ultrafeedback_dpo-no-sft_olmo_2024-04-02_15-30-49_858848)
python -u train.py model=olmo-7b datasets=[ultrafeedback] loss=dpo loss.beta=0.1 exp_name=ultrafeedback_dpo-no-sft_olmo gradient_accumulation_steps=8 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/home/tianhua3/agpt/agpt-dpo/state_dict/OLMo-7B.pt eval_every=12800 model.reference_dtype=bfloat16

## debug

python -u train.py model=olmo-7b datasets=[ultrafeedback] loss=sft exp_name=ultrafeedback_sft_olmo gradient_accumulation_steps=8 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=12800