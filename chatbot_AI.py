import tensorflow as tf
from transformers import pipeline, AutoTokenizer, TFAutoModelForSeq2SeqLM
import json

url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"

dataset_path = tf.keras.utils.get_file("databricks-dolly-15k.jsonl", url)

tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=512)
model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-base")

instruction_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
text_generation_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

with open(dataset_path, "r") as file:
    dataset = [json.loads(line) for line in file]

def find_matching_response(question):
    for data in dataset:
        if data["instruction"].lower() == question.lower():
            return data["response"]
    return None

while True:
    user_input = input("User: ")
    response = find_matching_response(user_input)
    if response is not None:
        print("Chatbot:", response)
    else:
        instruction = instruction_pipeline(user_input, truncation=True, max_length=512)[0]["generated_text"]
        response = text_generation_pipeline(instruction, max_length=50, do_sample=True)[0]["generated_text"]
        print("Chatbot:", response)
