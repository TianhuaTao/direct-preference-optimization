import transformers
from transformers import AutoModelForCausalLM

import torch

# model = AutoModelForCausalLM.from_pretrained("../models/Llama-2-7b-hf")
# model = AutoModelForCausalLM.from_pretrained("../models/pythia-2.8b")
model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-7B", trust_remote_code=True)

# target = '../state_dict/Llama-2-7b-hf.pt'
# target = '../state_dict/pythia-2.8b.pt'
target = '../state_dict/OLMo-7B.pt'

# save the state_dict
torch.save(
    {
            'step_idx': 0,
            'state': model.state_dict(),
            'metrics': {},
    }
,target)


