
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
import ast


TEST_FILE_PATH = 'val_data_direct_match_vs_eqn_classification_1455_2.pkl'
GOLD_FILE_PATH  = 'mtp_gold_trainClassifierWithout100_dm_vs_eqn_run2_FlanT5Large.txt'
OUTPUT_FILE_PATH = 'mtp_predictions_trainClassifierWithout100_dm_vs_eqn_run2_FlanT5Large.txt'

X_val = pickle.load(open(TEST_FILE_PATH,'rb'))
f1_metric = load("f1")


print(X_val)
# TEST_FILE_PATH = 'val_data_for_directMatching_composition_extraction_1039.pkl'
# GOLD_FILE_PATH = 'mtp_gold_CompExtractor_Without100_FlanT5Large_DirectMatch.txt'
# OUTPUT_FILE_PATH = 'mtp_predictions_CompExtractor_Without100_FlanT5Large_DirectMatch_run_2.txt'

def writeListToFile(lst, fname=OUTPUT_FILE_PATH):
    with open(fname, 'w', encoding='utf-8') as fp:
        for pred in lst:
            fp.write("%s\n" % pred)
    print(f"Written to file: {fname}")


with open(GOLD_FILE_PATH, 'r', encoding='utf-8') as fread:
        gold = fread.readlines()
with open(OUTPUT_FILE_PATH, 'r', encoding='utf-8') as fp:
        pred = fp.readlines()

result = f1_metric.compute(predictions=pred, references=gold)

print("F1 match score --->")
print(result)

gold_df = pd.DataFrame(gold, columns=['gold'])
pred_df = pd.DataFrame(pred, columns=['pred'])

extract_df = pd.concat([X_val,gold_df,pred_df], axis=1)

extract_df['pred'] = extract_df['pred'].apply(lambda x: x.replace('\n',''))
extract_df['gold'] = extract_df['gold'].apply(lambda x: x.replace('\n',''))

extract_df['pred'] = extract_df['pred'].apply(str)
extract_df['gold'] = extract_df['gold'].apply(str)

extract_df['pred'] = extract_df['pred'].apply(lambda x: ast.literal_eval(x))
extract_df['gold'] = extract_df['gold'].apply(lambda x: ast.literal_eval(x))

X_val_extract_SCC = extract_df[extract_df['pred'] == 0]

print(X_val_extract_SCC.head())


X_val_extract_SCC.rename(columns={'sentence':'Input', 'composition_gold':'Output'}, inplace=True)

print("------------------------------------")
print("Predicted as SCC = ",len(X_val_extract_SCC))
print("Predicted as MCC = ",len(X_val) - len(X_val_extract_SCC))
print("------------------------------------")


## EXTRACTION BEGINS HERE ------------------------------------------


# VAL_DATA_LOAD_PATH = 'val_data_for_directMatching_composition_extraction_1039.pkl'

TASK = 'CompExtractor_Without100_FlanT5Large_OnlyEqn'
RUNNUM = 'run_1'

TASK = 'CompExtractor_Without100_FlanT5Large_OnlyEqn_AdditionalGpt4Data'
RUNNUM = 'run_2'

TRAIN_DATA_LOAD_PATH = 'train_data_for_equation_composition_extraction_1744.pkl'
VAL_DATA_LOAD_PATH = 'val_data_for_equation_composition_extraction_416.pkl'


MODEL_SAVE_PATH = f'../scratch/mtp_{TASK}_{RUNNUM}.pt'
OUTPUT_FILE_PATH = f'mtp_predictions_{TASK}_{RUNNUM}.txt' 
GOLD_FILE_PATH = f'mtp_gold_{TASK}.txt'

# TASK = 'end2end_CompExtractor_Without100_FlanT5Large_DirectMatch'
# RUNNUM = 'run_1'

# MODEL_SAVE_PATH = f'../scratch/mtp_CompExtractor_Without100_FlanT5Large_DirectMatch_run_2.pt'
# OUTPUT_FILE_PATH = f'mtp_predictions_{TASK}_{RUNNUM}.txt' 
# GOLD_FILE_PATH = f'mtp_gold_{TASK}.txt'

MODEL_NAME = 'google/flan-t5-large'
# MODEL_LOAD_PATH = 'google/flan-t5-large'


max_input_length = 300
max_target_length = 800


exact_match_metric = load("exact_match")

X_val_extract_SCC['Output'] = X_val_extract_SCC['Output'].apply(str)

# def concatPrompt(df):
#     return str(df['Input']) + ' <SEP> ' + str(df['x']) + ' <SEP> ' + str(df['y']) + ' <SEP> ' + str(df['z'])
# X_val_extract['Input'] = X_val_extract.apply(lambda x: concatPrompt(x), axis=1)

X_val_extract = X_val_extract_SCC[['Input', 'Output']]


lst = X_val_extract['Output'].tolist()
writeListToFile(lst, GOLD_FILE_PATH)


device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_SAVE_PATH).to(device)


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
    writeListToFile(outputLst, OUTPUT_FILE_PATH)



print("Generating extraction predictions -->")
print(datetime.now())
generateTestOutput(model, tokenized_datasets_val, tokenizer)


print("Starting evaluations ----------->")

def evaluationScore(goldPath, predPath):
    with open(goldPath, 'r', encoding='utf-8') as fread:
        gold = fread.readlines()
    with open(predPath, 'r', encoding='utf-8') as fp:
        pred = fp.readlines()
    result = exact_match_metric.compute(predictions=pred, references=gold)
    return result

# print("Validation ended.. Calculating score -->")
# print(datetime.now())

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

