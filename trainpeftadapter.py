from peft import PeftModel, PeftConfig
from peftlorasetup import peft_model
from loadpretrainedmodel import tokenizer
from transformers import TrainingArguments, Trainer, AutoModelForSeq2SeqLM, AutoTokenizer
from finetuningwithpreprocessed import tokenized_datasets
import time
import torch
output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3,  # Higher learning rate than full fine-tuning.
    num_train_epochs=1,
    logging_steps=1,
    max_steps=1
)

peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
)

peft_trainer.train()

peft_model_path = "./peft-dialogue-summary-checkpoint-local"

peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

# it will take time to download we can use existing one to download with below command or i have already added the details in folder
# peft-dialogue-summary-checkpoint-from-s3
#!aws s3 cp --recursive s3://dlai-generative-ai/models/peft-dialogue-summary-checkpoint/ ./peft-dialogue-summary-checkpoint-from-s3/
#!ls -al ./peft-dialogue-summary-checkpoint-from-s3/adapter_model.bin

peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

peft_model = PeftModel.from_pretrained(peft_model_base,
                                       './peft-dialogue-summary-checkpoint-from-s3/',
                                       torch_dtype=torch.bfloat16,
                                       is_trainable=False)
