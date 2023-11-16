from loadpretrainedmodel import tokenizer
from pydataset import dataset


def tokenize_function(example):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue +
              end_prompt for dialogue in example["dialogue"]]
    example['input_ids'] = tokenizer(
        prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(
        example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids

    return example


# The dataset actually contains 3 diff splits: train, validation, test.
# The tokenize_function code is handling all data across all splits in batches.
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(
    ['id', 'topic', 'dialogue', 'summary',])

tokenized_datasets = tokenized_datasets.filter(
    lambda example, index: index % 100 == 0, with_indices=True)

print(f"Shapes of the datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")

print(tokenized_datasets)
