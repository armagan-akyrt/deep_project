import transformers
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import sentencepiece
import numpy as np
import os
import re
import argparse


model_checkpoint = "Helsinki-NLP/opus-mt-tr-en"

if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "translate Turkish to English: "
else:
    prefix = ""
    
max_input_length = 128
max_target_length = 128
source_lang = "tr"
target_lang = "en"

raw_datasets = load_dataset("wmt16", "tr-en")
    
    
metric = load_metric("sacrebleu")

print(raw_datasets["test"][0])

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

if "mbart" in model_checkpoint:
    tokenizer.tgt_lang = "en-XX"
    tokenizer.src_lang = "tr_TR"
        
        


def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs




def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result



def train_model(model_save_path, dataset_path, num_train_epoch):
    
    

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)



    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    batch_size = 16
    model_name = model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        f"{model_save_path}/{model_name}-finetuned-{source_lang}-to-{target_lang}",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_train_epoch,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    

    trainer.train()
    return trainer

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_save_path', default=os.getcwd(), type=str)
    parser.add_argument('--dataset_path', default="", type=str)
    parser.add_argument('--num_train_epoch', default=1, type=int)
    args = vars(parser.parse_args())
    print(f"Arguments {args}")
    train_model(model_save_path=args['model_save_path'], dataset_path=args['dataset_path'], num_train_epoch=args['num_train_epoch'])
    
main()
