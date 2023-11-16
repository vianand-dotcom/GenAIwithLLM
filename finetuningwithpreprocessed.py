from transformers import TrainingArguments, Trainer, AutoModelForSeq2SeqLM
from zeroshotinfering import original_model
from preprocessdataset import tokenized_datasets
import time
import torch
output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=1,
    max_steps=1
)

trainer = Trainer(
    model=original_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)
# trainer.train()

# training a fully fine-tuned model will take a lot of time to utilizie the existing pretrained model we can use aws sagmaker
#!aws s3 cp --recursive s3://dlai-generative-ai/models/flan-dialogue-summary-checkpoint/ ./flan-dialogue-summary-checkpoint/
#!ls -alh ./flan-dialogue-summary-checkpoint/pytorch_model.bin

# instruct_model = AutoModelForSeq2SeqLM.from_pretrained(
# "./flan-dialogue-summary-checkpoint", torch_dtype=torch.bfloat16)
