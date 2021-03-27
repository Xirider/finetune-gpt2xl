from transformers import GPTNeoForCausalLM, GPTNeoTokenizer
model = GPTNeoForCausalLM.from_pretrained("finetunednowarm").to("cuda")
tokenizer = GPTNeoTokenizer.from_pretrained("finetunednowarm")
text = "When"
ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
max_length = 400 + ids.shape[1]
do_sample = True
gen_tokens = model.generate(ids, do_sample=do_sample, top_p=0.9, temperature=1.0,
                            min_length=max_length, max_length=max_length, use_cache=False)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)
