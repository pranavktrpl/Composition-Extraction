import pandas as pd
import numpy as np
from datetime import datetime
import torch
import os
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from evaluate import load
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer#,DataCollatorForSeq2Seq, 
from torch import cuda
import pickle

# # WandB – Import the wandb library
import wandb

CLASSIFIER_VAL_DATA_LOAD_PATH = 'val_data_direct_match_vs_eqn_classification_1455.pkl'


RUNNUM = 1
RatioRunName = f'dm_vs_eqn_run{RUNNUM}'
TASK = 'dm_vs_eqn'

OUTPUT_FILE_PATH = f'mtp_predictions_end2end_trainClassifierWithout100_{RatioRunName}_FlanT5Large.txt' 
GOLD_FILE_PATH = f'mtp_gold_end2end_trainClassifierWithout100_{RatioRunName}_FlanT5Large.txt'


MODEL_NAME = 'google/flan-t5-large'
MODEL_LOAD_PATH = f'../scratch/mtp_trainClassifierWithout100_{RatioRunName}_FlanT5Large.pt'


f1_metric = load("f1")

max_input_length = 700
max_target_length = 10


def writeListToFile(lst, fname=OUTPUT_FILE_PATH):
    with open(fname, 'w', encoding='utf-8') as fp:
        for pred in lst:
            fp.write("%s\n" % pred)
    print(f"Written to file: {fname}")


X_val = pickle.load(open(CLASSIFIER_VAL_DATA_LOAD_PATH,'rb'))
X_val.rename(columns = {'sentence':'Input', 'is_SCC':'Output'}, inplace=True)
X_val = X_val[["Input", "Output"]]


lst = X_val['Output'].tolist()
writeListToFile(lst, GOLD_FILE_PATH)



device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_LOAD_PATH).to(device)


def preprocessTrainDev(examples):
    inputs = [doc for doc in examples["Input"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length ,padding='max_length')
    
    
    outputs = [str(doc) for doc in examples["Output"]]
    labels = tokenizer(outputs, max_length=max_target_length, padding='max_length')
    
    labelsInpIds = labels.input_ids
    
    labelsInpIds = np.array(labelsInpIds)
    labelsInpIds[labelsInpIds == tokenizer.pad_token_id] = -100
    labelsInpIds = list(labelsInpIds)
    model_inputs["labels"] = labelsInpIds

    return model_inputs


datasetVal = Dataset.from_pandas(X_val)
tokenized_datasets_val = datasetVal.map(preprocessTrainDev, batched=True)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = f1_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return result

def generateTestOutput(model, testdata, tokenizer):
    model.eval()
    data = testdata['input_ids']
    amask = testdata['attention_mask']
    ds = Dataset.from_dict({"data":data, "attention_mask":amask}).with_format("torch")
    dataloader = DataLoader(ds, batch_size=1, shuffle=False)
    outputLst =[]
    # total_loss = 0.0
    
    with torch.no_grad():
        for i, (inputs) in enumerate(tqdm(dataloader)):
            input_ids = inputs['data'].to(device)
            amasks = inputs['attention_mask'].to(device)
            # generate model outputs
            generated_ids = model.generate(
                input_ids = input_ids,
                attention_mask = amasks,
                max_new_tokens=10,
                num_beams = 2)
            # print(generated_ids)
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            outputLst.extend(output)
    writeListToFile(outputLst)


print(datetime.now())
print("------------------- Generating prediction file ------------------")
print(datetime.now())
generateTestOutput(model, tokenized_datasets_val, tokenizer)



print("------------------- Evaluating Scores ------------------")

def evaluationScore(goldPath, predPath):
    with open(goldPath, 'r', encoding='utf-8') as fread:
        gold = fread.readlines()
    with open(predPath, 'r', encoding='utf-8') as fp:
        pred = fp.readlines()
    result = f1_metric.compute(predictions=pred, references=gold)
    return result

# print("Validation ended.. Calculating score -->")

print("F1 match score --->")
print(evaluationScore(GOLD_FILE_PATH, OUTPUT_FILE_PATH))


