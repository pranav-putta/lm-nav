import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/Llama-2-7B-32K-Instruct")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/Llama-2-7B-32K-Instruct",
                                             trust_remote_code=True, torch_dtype=torch.float16)
input_ids = tokenizer.encode("[INST]\nWrite a poem about cats\n[/INST]\n\n", return_tensors="pt")
output = model.generate(input_ids, max_length=128,
                        temperature=0.7, repetition_penalty=1.1, top_p=0.7, top_k=50)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)