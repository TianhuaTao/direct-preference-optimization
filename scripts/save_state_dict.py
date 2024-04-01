import transformers
from transformers import AutoModelForCausalLM

import torch

model = AutoModelForCausalLM.from_pretrained("/home/tianhua3/agpt/agpt-dpo/models/Llama-2-7b-hf")

target = '/home/tianhua3/agpt/agpt-dpo/state_dict/Llama-2-7b-hf.pt'


# save the state_dict
torch.save(model.state_dict(), target)


