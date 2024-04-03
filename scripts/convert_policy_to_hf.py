import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

all_models = [
    # hf_model_name, policy_state_dict, output_dir
    ["models/Llama-2-7b-hf", 'direct-preference-optimization/.cache/tianhua3/ultrafeedback_sft_llama2_2024-04-01_01-30-03_581140/LATEST/policy.pt', 'outputs/ultrafeedback_sft_llama2'],
    ["models/Llama-2-7b-hf", 'direct-preference-optimization/.cache/tianhua3/ultrafeedback_dpo_llama2_2024-04-01_02-05-30_347982/LATEST/policy.pt', 'outputs/ultrafeedback_dpo_llama2'],
    ["models/Llama-2-7b-hf", 'direct-preference-optimization/.cache/tianhua3/ultrafeedback_dpo-no-sft_llama2_2024-04-01_13-11-00_220708/LATEST/policy.pt', 'outputs/ultrafeedback_dpo-no-sft_llama2'],

    ["models/Llama-2-7b-hf", 'direct-preference-optimization/.cache/tianhua3/hh_sft_llama2_2024-04-01_08-19-32_314096/LATEST/policy.pt', 'outputs/hh_sft_llama2'],
    ["models/Llama-2-7b-hf", 'direct-preference-optimization/.cache/tianhua3/hh_dpo_llama2_2024-04-01_06-38-44_189131/LATEST/policy.pt', 'outputs/hh_dpo_llama2'],
    ["models/Llama-2-7b-hf", 'direct-preference-optimization/.cache/tianhua3/hh_dpo-no-sft_llama2_2024-04-01_16-06-42_761861/LATEST/policy.pt', 'outputs/hh_dpo-no-sft_llama2'],
    
    ["models/pythia-2.8b", 'direct-preference-optimization/.cache/tianhua3/hh_sft_pythia28_2024-03-31_20-50-15_031216/LATEST/policy.pt', 'outputs/hh_sft_pythia28'],
    ["models/pythia-2.8b", 'direct-preference-optimization/.cache/tianhua3/hh_dpo_pythia28_2024-03-31_21-20-06_233890/LATEST/policy.pt', 'outputs/hh_dpo_pythia28'],
    ["models/pythia-2.8b", 'direct-preference-optimization/.cache/tianhua3/hh_dpo-no-sft_pythia28_2024-04-03_05-08-08_902418/LATEST/policy.pt', 'outputs/hh_dpo-no-sft_pythia28'],
    
    

    ["models/pythia-2.8b", 'direct-preference-optimization/.cache/tianhua3/ultrafeedback_sft_pythia28_2024-04-02_20-40-52_114340/LATEST/policy.pt', 'outputs/ultrafeedback_sft_pythia28'],
    ["models/pythia-2.8b", 'direct-preference-optimization/.cache/tianhua3/ultrafeedback_dpo_pythia28_2024-04-02_20-57-07_074166/LATEST/policy.pt', 'outputs/ultrafeedback_dpo_pythia28'],
    ["models/pythia-2.8b", 'direct-preference-optimization/.cache/tianhua3/ultrafeedback_dpo-no-sft_pythia28_2024-04-02_22-32-13_845112/LATEST/policy.pt', 'outputs/ultrafeedback_dpo-no-sft_pythia28'],


    ["models/OLMo-7B", 'direct-preference-optimization/.cache/tianhua3/ultrafeedback_sft_olmo_2024-04-02_13-25-27_040353/LATEST/policy.pt', 'outputs/ultrafeedback_sft_olmo'],
    ["models/OLMo-7B", 'direct-preference-optimization/.cache/tianhua3/ultrafeedback_dpo_olmo_2024-04-02_14-06-12_855447/LATEST/policy.pt', 'outputs/ultrafeedback_dpo_olmo'],
    ["models/OLMo-7B", 'direct-preference-optimization/.cache/tianhua3/ultrafeedback_dpo-no-sft_olmo_2024-04-02_15-30-49_858848/LATEST/policy.pt', 'outputs/ultrafeedback_dpo-no-sft_olmo'],
    
    # ["models/OLMo-7B", None, None],
]

for hf_model_name, state_dict, output_dir in all_models:
    # skip if the target exists
    if os.path.exists(output_dir):
        print(f'{output_dir} exists, skipping')
        continue

    # init empty model
    print(f'init model {hf_model_name}')
    model = AutoModelForCausalLM.from_pretrained(hf_model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)

    #load the state_dict
    print(f'load state_dict {state_dict}')
    state_dict = torch.load(state_dict, map_location='cpu')

    # remove "*/rotary_emb.inv_freq" in the state_dict
    for key in list(state_dict['state'].keys()):
        if ".rotary_emb.inv_freq" in key:
            del state_dict['state'][key]

    # update the state_dict
    print(f'update state_dict')
    model.load_state_dict(state_dict['state'])
    # fp16
    model.half()

    # save the model
    print(f'save model {output_dir}')
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


