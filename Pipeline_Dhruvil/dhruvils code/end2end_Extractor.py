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
import pdb

# # WandB – Import the wandb library
import wandb

MODEL_SAVE_PATH = '../scratch/mtp_trainEnd2EndExtractor_run_1.pt'
OUTPUT_FILE_PATH = 'mtp_predictions_trainEnd2EndExtractor_run_1.txt' 
GOLD_FILE_PATH = 'mtp_gold_trainEnd2EndExtractor.txt'
# # %env WANDB_PROJECT = 'NLPA3_HPC'
# os.environ["WANDB_WATCH"] : all
# os.environ["WANDB_PROJECT"] : 'NLPA3_HPC'

# MODEL_LOAD_PATH = "ConvLab/t5-small-dst-multiwoz21_sgd_tm1_tm2_tm3"
MODEL_NAME = 't5-base'
MODEL_LOAD_PATH = 't5-base'
max_input_length = 300
max_target_length = 800
batch_size = 4
LR=1e-3
WEIGHT_DECAY=0.01
EPOCHS=30
# prefix = "parse: "

run = wandb.init(
    # Set the project where this run will be logged
    project="MTP_T5_End2End_Extractor",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 1e-3,
        "epochs": 20,
        "weight_decay": 0.01,
        "max_length":800
    },
    name='end2end_epoch20')

# WandBRunName = "text_comp_extractor_run_2_testrun"

exact_match_metric = load("exact_match")


def writeListToFile(lst, fname=OUTPUT_FILE_PATH):
    with open(fname, 'w', encoding='utf-8') as fp:
        for pred in lst:
            fp.write("%s\n" % pred)
    print(f"Written to file: {fname}")




X_train = pickle.load(open('endToEndDirectCompositionExtractionData_Ratio1to5.pkl','rb'))
X_val = pickle.load(open('val_data_combined_end2end_329papers.pkl','rb'))


# X_val = pickle.load(open('X_val_compExtractor.pkl','rb'))
# y_train = pickle.load(open('y_train_compExtractor.pkl','rb'))
# y_val = pickle.load(open('y_val_compExtractor.pkl','rb'))

# X_train = X_train[::100]
# X_val = X_val[::100]
# y_train = y_train[::100]
# y_val = y_val[::100]

# print("###########")
# print(y_train)
# print("------------")
# print(y_val)


# X_train.rename(columns = {'sentence':'Input'}, inplace=True)
# X_val.rename(columns = {'sentence':'Input'}, inplace=True)
# y_train.rename(columns = {'Ig_comp':'Output'}, inplace=True)
# y_val.rename(columns = {'Ig_comp':'Output'}, inplace=True)


# y_train['Output'] = y_train['Output'].apply(str)
# y_val['Output'] = y_val['Output'].apply(str)


# X_train = pd.concat([X_train, y_train], axis=1)
# X_val = pd.concat([X_val, y_val], axis=1)

# X_train = pd.concat([X_train, X_val], axis=0)

# X_val = pickle.load(open('val_CompositionForExtraction_dhruvil_1744.pkl','rb'))
# X_val.rename(columns = {'composition':'Output'}, inplace=True)
X_train['Output'] = X_train['Output'].apply(str)
X_val['Output'] = X_val['Output'].apply(str)


# y_train = pd.DataFrame(X_train).rename(columns={'comp':'Output'})
# X_val = pd.DataFrame(X_val).rename(columns={'comp':'Output'})

# def parseOutput(df):
#     return ast.literal_eval(df['Output'])


def concatPrompt(df):
    return str(df['Input']) + ' <SEP> ' + str(df['x']) + ' <SEP> ' + str(df['y']) + ' <SEP> ' + str(df['z'])

# X_val['Output2'] = X_val.apply(lambda x: parseOutput(x), axis=1)

X_train['Input'] = X_train.apply(lambda x: concatPrompt(x), axis=1)
X_val['Input'] = X_val.apply(lambda x: concatPrompt(x), axis=1)


print(X_val)

X_train = X_train[['Input', 'Output']]
X_val = X_val[['Input', 'Output']]

print(X_val)

lst = X_val['Output'].tolist()
writeListToFile(lst, GOLD_FILE_PATH)



device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
# tokenizer.add_tokens(['<SEP>'])

model = T5ForConditionalGeneration.from_pretrained(MODEL_LOAD_PATH).to(device)


def preprocessTrainDev(examples):
    inputs = [doc for doc in examples["Input"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,add_special_tokens=True ,padding='max_length')
    
    # contexts = tokenizer(examples["Context"], max_length=max_input_length, truncation=True,add_special_tokens=True,padding='max_length')
    # model_inputs["contexts"] = contexts
    print("LABELS")
    
    # outputs = [str(doc) for doc in examples["Output"]]
    labels = tokenizer(examples["Output"], max_length=max_target_length, truncation=True,add_special_tokens=True, padding='max_length')
    # print(labels.shape)
    # print(labels)
    labelsInpIds = labels.input_ids
    # print(labelsInpIds)
    labelsInpIds = np.array(labelsInpIds)
    labelsInpIds[labelsInpIds == tokenizer.pad_token_id] = -100
    labelsInpIds = list(labelsInpIds)
    model_inputs["labels"] = labelsInpIds

    return model_inputs

# def preprocessTest(examples):
#     inputs = [doc for doc in examples["Input"]]
#     model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,add_special_tokens=True,padding='max_length')

#     return model_inputs




# X_train['Output'] = X_train['Output'].astype(str)
# X_val['Output'] = X_val['Output'].astype(str)

datasetTrain = Dataset.from_pandas(X_train)
datasetVal = Dataset.from_pandas(X_val)
# datasetTest = Dataset.from_pandas(test_df)

tokenized_datasets_train = datasetTrain.map(preprocessTrainDev, batched=True)
tokenized_datasets_val = datasetVal.map(preprocessTrainDev, batched=True)
# tokenized_datasets_test = datasetTest.map(preprocessTest, batched=True)

model_name = "MTP_ExtractEnd2EndComp"
args = Seq2SeqTrainingArguments(
    f"../scratch/{model_name}-End2End_run1",
    evaluation_strategy = "epoch",
    learning_rate=1e-3,
    # logging_steps=1,
    # per_device_train_batch_size=4,
    # per_device_eval_batch_size=4,
    weight_decay=0.01,
    #save_total_limit=3,
    num_train_epochs=20,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,
    generation_max_length=800,
    generation_num_beams=1,  
    save_total_limit = 2, 
    metric_for_best_model = 'exact_match',   
    greater_is_better = True, 
    load_best_model_at_end = True,
    save_strategy = "epoch",
    report_to = "wandb",
    # run_name = "text_comp_extract_run_30epochs_4_newVal"
)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # print("PREDS")
    # print(preds)
    # print("LABELS")
    # print(labels)
    # decoded_preds = []
    # pd.DataFrame(preds).to_csv('predsIndexes.csv')
    # pdb.set_trace()
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    # for i in range (preds.shape[0]):
    #     try:
    #         decoded_preds.append( tokenizer.decode(preds[i], skip_special_tokens=True) )
    #     except:
    #         print(i)
    # print("Decoded preds")
    
    decoded_preds =  tokenizer.batch_decode(preds, skip_special_tokens=True) 
    # print(decoded_preds)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # print(decoded_labels)
    result = exact_match_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return result

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets_train,
    eval_dataset=tokenized_datasets_val,
    #data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Starting Training -->")
print(datetime.now())

best_model = trainer.train()

print("Training ended -->")
print(datetime.now())

trainer.save_model(MODEL_SAVE_PATH)


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

print("Validating from generate -->")
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
