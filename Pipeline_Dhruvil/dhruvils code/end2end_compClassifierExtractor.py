import jsonlines
import pandas as pd
import random
# from datasets import load_dataset, load_metric

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
import ast
# # WandB – Import the wandb library
import wandb

MODEL_SAVE_PATH = '../scratch/mtp_trainClassifier_run_ratio1to4_2.pt'
OUTPUT_FILE_PATH = 'mtp_predictions_trainClassifier_ratio1to4_2.txt' 
GOLD_FILE_PATH = 'mtp_gold_trainClassifier_ratio1to4_2.txt'

MODEL_NAME = 't5-base'
MODEL_LOAD_PATH = 't5-base'
max_input_length = 700
max_target_length = 10
batch_size = 4
LR=1e-3,
WEIGHT_DECAY=0.01,
EPOCHS=10

f1_metric = load("f1")

def writeListToFile(lst, fname=OUTPUT_FILE_PATH):
    with open(fname, 'w', encoding='utf-8') as fp:
        for pred in lst:
            fp.write("%s\n" % pred)
    print(f"Written to file: {fname}")


# X_train = pickle.load(open('train_data_ForCompositionClassification_Ratio1to2.pkl','rb'))
X_val = pickle.load(open('val_data_ForCompositionClassification_42875Sent_330Papers.pkl','rb'))
X_val_classify = X_val.rename(columns={'sentence':'Input', 'is_composition':'Output'})



device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
# model = T5ForConditionalGeneration.from_pretrained(MODEL_LOAD_PATH).to(device)


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

# datasetTrain = Dataset.from_pandas(X_train)
# tokenized_datasets_train = datasetTrain.map(preprocessTrainDev, batched=True)

datasetVal = Dataset.from_pandas(X_val_classify)
tokenized_datasets_val = datasetVal.map(preprocessTrainDev, batched=True)


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
                num_beams = 1)
            # print(generated_ids)
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            outputLst.extend(output)
    writeListToFile(outputLst)


model = T5ForConditionalGeneration.from_pretrained(MODEL_SAVE_PATH).to(device)

print("Generating classification predictions -->")
print(datetime.now())

generateTestOutput(model, tokenized_datasets_val, tokenizer)


print("GENERATION COMPLETED !!")

print("Calculating score -->")
print(datetime.now())


with open(GOLD_FILE_PATH, 'r', encoding='utf-8') as fread:
        gold = fread.readlines()
with open(OUTPUT_FILE_PATH, 'r', encoding='utf-8') as fp:
        pred = fp.readlines()

result = f1_metric.compute(predictions=pred, references=gold)

print("F1 match score --->")
print(result)

gold_df = pd.DataFrame(gold, columns=['Gold'])
pred_df = pd.DataFrame(pred, columns=['Pred'])

extract_df = pd.concat([X_val,gold_df,pred_df], axis=1)

extract_df['pred'] = extract_df['pred'].apply(lambda x: x.replace('\n',''))
extract_df['gold'] = extract_df['gold'].apply(lambda x: x.replace('\n',''))

extract_df['pred'] = extract_df['pred'].apply(str)
extract_df['gold'] = extract_df['gold'].apply(str)

extract_df['pred'] = extract_df['pred'].apply(lambda x: ast.literal_eval(x))
extract_df['gold'] = extract_df['gold'].apply(lambda x: ast.literal_eval(x))

X_val_extract = extract_df[extract_df['pred'] == '1']
X_val_extract.rename(columns={'sentence':'Input', 'composition':'Output'}, inplace=True)


## EXTRACTION BEGINS HERE ------------------------------------------

MODEL_SAVE_PATH = '../scratch/mtp_trainEnd2EndExtractor_run_1.pt'
OUTPUT_FILE_PATH = 'mtp_predictions_trainEnd2EndExtractor_run_1.txt' 
GOLD_FILE_PATH = 'mtp_gold_trainEnd2EndExtractor.txt'

MODEL_NAME = 't5-base'
MODEL_LOAD_PATH = 't5-base'
max_input_length = 300
max_target_length = 800
batch_size = 4
LR=1e-3
WEIGHT_DECAY=0.01
EPOCHS=30


exact_match_metric = load("exact_match")


def writeListToFile(lst, fname=OUTPUT_FILE_PATH):
    with open(fname, 'w', encoding='utf-8') as fp:
        for pred in lst:
            fp.write("%s\n" % pred)
    print(f"Written to file: {fname}")

X_val_extract['Output'] = X_val_extract['Output'].apply(str)

def concatPrompt(df):
    return str(df['Input']) + ' <SEP> ' + str(df['x']) + ' <SEP> ' + str(df['y']) + ' <SEP> ' + str(df['z'])

X_val_extract['Input'] = X_val_extract.apply(lambda x: concatPrompt(x), axis=1)

X_val_extract = X_val_extract[['Input', 'Output']]


lst = X_val['Output'].tolist()
writeListToFile(lst, GOLD_FILE_PATH)


device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)


def preprocessTrainDev(examples):
    inputs = [doc for doc in examples["Input"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,add_special_tokens=True ,padding='max_length')
    labels = tokenizer(examples["Output"], max_length=max_target_length, truncation=True,add_special_tokens=True, padding='max_length')
    labelsInpIds = labels.input_ids
    
    labelsInpIds = np.array(labelsInpIds)
    labelsInpIds[labelsInpIds == tokenizer.pad_token_id] = -100
    labelsInpIds = list(labelsInpIds)
    model_inputs["labels"] = labelsInpIds

    return model_inputs

datasetVal = Dataset.from_pandas(X_val_extract)
tokenized_datasets_val = datasetVal.map(preprocessTrainDev, batched=True)


def generateTestOutput(model, testdata, tokenizer):
    model.eval()
    data = testdata['input_ids']
    amask = testdata['attention_mask']
    ds = Dataset.from_dict({"data":data, "attention_mask":amask}).with_format("torch")
    dataloader = DataLoader(ds, batch_size=8, shuffle=False)
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
                max_new_tokens=800,
                num_beams = 1)
            # print(generated_ids)
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            outputLst.extend(output)
    writeListToFile(outputLst)


model = T5ForConditionalGeneration.from_pretrained(MODEL_SAVE_PATH).to(device)

print("Generating extraction predictions -->")
print(datetime.now())
generateTestOutput(model, tokenized_datasets_val, tokenizer)

def evaluationScore(goldPath, predPath):
    with open(goldPath, 'r', encoding='utf-8') as fread:
        gold = fread.readlines()
    with open(predPath, 'r', encoding='utf-8') as fp:
        pred = fp.readlines()
    result = exact_match_metric.compute(predictions=pred, references=gold)
    return result

print("Validation ended.. Calculating score -->")
print(datetime.now())

print("Exact match score --->")
print(evaluationScore(GOLD_FILE_PATH, OUTPUT_FILE_PATH))

# EVALUATION 2 ----------->
def evaluationScore(goldPath, predPath):
    with open(goldPath, 'r', encoding='utf-8') as fread:
        gold = fread.readlines()
    with open(predPath, 'r', encoding='utf-8') as fp:
        pred = fp.readlines()
    
    result = 0
    ct=0
    for g,p in zip(gold, pred):
        # print("GOLD")
        # print(g)
        # print("PRED")
        # print(p)
        evalscore = evaluateComposition(g, p)
        # print(evalscore)
        result += evalscore
        # print(result)
        ct+=1
    # result = exact_match_metric.compute(predictions=pred, references=gold)
    return result/ct

def evaluateComposition(gold, predicted):
    
    gold = ast.literal_eval(gold)
    try:
        predicted = ast.literal_eval(predicted)
    except:
        # print("Cannot parse")
        # print(predicted)
        return 0.0
    gold_sets = [set(compound_list) for compound_list in gold]
    predicted_sets = [set(compound_list) for compound_list in predicted]
    
    # print("INSIDE EC")
    # print(gold_sets)
    # print(predicted_sets)
    # print("CLOSE")
    
    precision=0
    recall=0

    for predicted_set in predicted_sets:
        if(predicted_set in gold_sets):
            precision+=1
    
    for gold_set in gold_sets:
        if(gold_set in predicted_sets):
            recall+=1   
    
    precision /= len(predicted_sets)
    recall /= len(gold_sets)
    
    if(precision==0 and recall ==0):
        f1=0.0
    else: 
        f1 = (2*precision*recall)/(precision+recall)
    
    return round(f1,2)


print("Metric score --->")
print(evaluationScore(GOLD_FILE_PATH, OUTPUT_FILE_PATH))


with open(OUTPUT_FILE_PATH, 'r', encoding='utf-8') as fp:
    pred = fp.readlines()


pred_extract_df = pd.DataFrame(pred, columns=['PredExtract'])
extract_df['PredExtract'] = extract_df['PredExtract'].apply(lambda x: x.replace('\n',''))
extract_df['PredExtract'] = extract_df['PredExtract'].apply(str)
extract_df['PredExtract'] = extract_df['PredExtract'].apply(lambda x: ast.literal_eval(x))


# final_df = pd.concat([X_val, pred_extract_df], axis=1)
X_val_extracted = extract_df[extract_df['pred'] == '1']
pred_combined = pd.concat([X_val, pred_extract_df], axis=1)
X_val_remaining = extract_df[extract_df['pred'] == '0']

final_df = pd.concat([X_val, pred_extract_df], axis=0)


def final_evaluation(goldLst, predLst):
    result = 0
    ct=0
    for g,p in zip(goldLst, predLst):
        # print("GOLD")
        # print(g)
        # print("PRED")
        # print(p)
        evalscore = evaluateComposition(g, p)
        # print(evalscore)
        result += evalscore
        # print(result)
        ct+=1
    # result = exact_match_metric.compute(predictions=pred, references=gold)
    return result/ct

print("FINAL RESULT --->>>")
print(final_evaluation(final_df['composition'].tolist(), final_df['PredExtract'].tolist()))

