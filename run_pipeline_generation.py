from transformers import pipeline

generator = pipeline('text-generation', model='finetuned')
generated_text = generator("This", max_length=200)[0]["generated_text"]
print(generated_text)
