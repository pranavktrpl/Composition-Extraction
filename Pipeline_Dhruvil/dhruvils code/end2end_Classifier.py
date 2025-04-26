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

NEGATIVE_TO_POSITIVE_RATIO = 10
RUN_NUM = 1

CLASSIFIER_TRAIN_DATA_LOAD_PATH = f'train_data_ForCompositionClassification_Ratio1to{NEGATIVE_TO_POSITIVE_RATIO}.pkl'
CLASSIFIER_VAL_DATA_LOAD_PATH = 'val_data_ForCompositionClassification_42875Sent_330Papers.pkl'


RatioRunName = f'ratio1to{NEGATIVE_TO_POSITIVE_RATIO}_run{RUN_NUM}'



MODEL_SAVE_PATH = f'../scratch/mtp_trainClassifier_{RatioRunName}.pt'
OUTPUT_FILE_PATH = f'mtp_predictions_trainClassifier_{RatioRunName}.txt' 
GOLD_FILE_PATH = f'mtp_gold_trainClassifier_ratio1to{NEGATIVE_TO_POSITIVE_RATIO}.txt'

WandBRunName = f"compClassifier_{RatioRunName}"


MODEL_NAME = 't5-base'
MODEL_LOAD_PATH = 't5-base'
max_input_length = 700
max_target_length = 10
batch_size = 4
LR=1e-3,
WEIGHT_DECAY=0.01,
EPOCHS=10


run = wandb.init(
    # Set the project where this run will be logged
    project="MTP_T5_End2End_Classifier",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 1e-3,
        "epochs": 10,
        "weight_decay": 0.01,
        "max_input_length": 700,
    },
    name = WandBRunName)


f1_metric = load("f1")


def writeListToFile(lst, fname=OUTPUT_FILE_PATH):
    with open(fname, 'w', encoding='utf-8') as fp:
        for pred in lst:
            fp.write("%s\n" % pred)
    print(f"Written to file: {fname}")


X_train = pickle.load(open(CLASSIFIER_TRAIN_DATA_LOAD_PATH,'rb'))
X_val = pickle.load(open(CLASSIFIER_VAL_DATA_LOAD_PATH,'rb'))

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

# def preprocessTest(examples):
#     inputs = [doc for doc in examples["Input"]]
#     model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,add_special_tokens=True,padding='max_length')

#     return model_inputs


datasetTrain = Dataset.from_pandas(X_train)
datasetVal = Dataset.from_pandas(X_val)
# datasetTest = Dataset.from_pandas(test_df)

tokenized_datasets_train = datasetTrain.map(preprocessTrainDev, batched=True)
tokenized_datasets_val = datasetVal.map(preprocessTrainDev, batched=True)
# tokenized_datasets_test = datasetTest.map(preprocessTest, batched=True)

model_name = "MTP_T5_End2End_Classifier"
args = Seq2SeqTrainingArguments(
    f"../scratch/{model_name}-{RatioRunName}",
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
    generation_max_length=10,
    generation_num_beams=2,  
    save_total_limit = 2, 
    metric_for_best_model = 'f1',   
    greater_is_better = True, 
    load_best_model_at_end = True,
    save_strategy = "epoch",
    report_to = "wandb",
    # run_name = "text_comp_classify_run_ratio1to5_2"
)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = f1_metric.compute(predictions=decoded_preds, references=decoded_labels)
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


model = T5ForConditionalGeneration.from_pretrained(MODEL_SAVE_PATH).to(device)

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
