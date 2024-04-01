import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

hf_model_name = "models/Llama-2-7b-hf"
# hf_model_name = "models/OLMo-7B"
# hf_model_name = "models/pythia-2.8b"


state_dict = 'direct-preference-optimization/.cache/tianhua3/ultrafeedback_sft_llama2_2024-04-01_01-30-03_581140/LATEST/policy.pt'

output_dir = 'outputs/ultrafeedback_sft_llama2'

# init empty model
print('init model')
model = AutoModelForCausalLM.from_pretrained(hf_model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)

#load the state_dict
print('load state_dict')
state_dict = torch.load(state_dict, map_location='cpu')
model.load_state_dict(state_dict['state'])
# fp16
model.half()

# save the model
print('save model')
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


