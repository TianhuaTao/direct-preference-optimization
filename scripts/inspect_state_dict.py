import torch

# state_dict = torch.load('/home/tianhua3/agpt/agpt-dpo/direct-preference-optimization/.cache/tianhua3/ultrafeedback_sft_llama2_2024-03-31_20-15-30_309905/LATEST/policy.pt')
state_dict2 = torch.load('/home/tianhua3/agpt/agpt-dpo/state_dict/Llama-2-7b-hf.pt')

# print(state_dict.keys())
print(state_dict2.keys())