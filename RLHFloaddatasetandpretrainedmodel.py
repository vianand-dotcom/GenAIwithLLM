from datasets import load_dataset
model_name = "google/flan-t5-base"
huggingface_dataset_name = "knkarthick/dialogsum"

dataset_original = load_dataset(huggingface_dataset_name)

print(dataset_original)
