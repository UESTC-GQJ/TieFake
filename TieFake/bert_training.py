import sys
import os
import time
import datetime
import pandas as pd
import numpy as np
import json, re
import random
from tqdm import tqdm_notebook
from uuid import uuid4
from sklearn.metrics import matthews_corrcoef, confusion_matrix,accuracy_score,f1_score

## Torch Modules
import torch,gc
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import RandomSampler, SequentialSampler
import json
import codecs
# loading pre-trained models
from transformers import get_linear_schedule_with_warmup
from transformers import (
    BertForSequenceClassification,
#     TFBertForSequenceClassification, 
                          BertTokenizer,
#                           TFRobertaForSequenceClassification,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                         AdamW)

import logging
logging.basicConfig(level = logging.ERROR)
import ssl
from numpyencoder import NumpyEncoder
gc.collect()
torch.cuda.empty_cache()
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

if torch.cuda.is_available():      
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def tokenize_dataset(df, num_of_way):
    df = df.sample(frac=1).reset_index(drop=True)    

    print('Number of training sentences: {:,}\n'.format(df.shape[0]))     
    sentences = df['clean_title'].values
    labels = df['label'].values

    print(' Original: ', sentences[0])    
    print('Tokenized BERT: ', bert_tokenizer.tokenize(sentences[0]))    
    print('Token IDs BERT: ', bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(sentences[0])))

    max_len_bert = 0

    for sent in sentences:    
        input_ids_bert = bert_tokenizer.encode(sent, add_special_tokens=True)       
        max_len_bert = max(max_len_bert, len(input_ids_bert))

    print('Max sentence length BERT: ', max_len_bert)    
    bert_input_ids = []
    bert_attention_masks = []
    sentence_ids = []
    counter = 0

    for sent in sentences:
        bert_encoded_dict = bert_tokenizer.encode_plus(
                            sent,                      
                            add_special_tokens = True, 
                            max_length = 512,          
                            pad_to_max_length = True,
                            return_attention_mask = True, 
                            return_tensors = 'pt',   
                       ) 
        bert_input_ids.append(bert_encoded_dict['input_ids'])    
        bert_attention_masks.append(bert_encoded_dict['attention_mask'])
        sentence_ids.append(counter)
        counter  = counter + 1

    bert_input_ids = torch.cat(bert_input_ids, dim=0)
    bert_attention_masks = torch.cat(bert_attention_masks, dim=0)      

    labels = torch.tensor(labels)
    sentence_ids = torch.tensor(sentence_ids)
    torch.manual_seed(0)
    bert_dataset = TensorDataset(sentence_ids, bert_input_ids, bert_attention_masks, labels)
    return bert_dataset

def index_remover(tensordata):
    input_ids = []
    attention_masks = []
    labels = []
   
    for a,b,c,d in tensordata:
        input_ids.append(b.tolist())
        attention_masks.append(c.tolist())
        labels.append(d.tolist())
        
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)
    
    final_dataset =  TensorDataset(input_ids, attention_masks, labels)
    return final_dataset

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))

    return str(datetime.timedelta(seconds=elapsed_rounded))

df_train = pd.read_csv('./Data/gossipcop_train.tsv',encoding='ansi',delimiter="\t")
df_test = pd.read_csv('./Data/gossipcop_test.tsv',encoding='ansi',delimiter="\t")
df_train = df_train[df_train['clean_text'].notna()]
df_test = df_test[df_test['clean_text'].notna()]

num_of_way = 2 

bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                                num_labels = num_of_way, 
                                                                output_attentions = False, 
                                                                output_hidden_states = False 
                                                          )

bert_model.to(device)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print(' BERT model loaded')
bert_train_dataset = tokenize_dataset(df_train,num_of_way)
bert_train_dataset = index_remover(bert_train_dataset)
batch_size = 16
bert_train_dataloader = DataLoader(
            bert_train_dataset,  
            sampler = RandomSampler(bert_train_dataset), 
            batch_size = batch_size
        )
bert_optimizer = AdamW(bert_model.parameters(),
                  lr = 5e-5, 
                  eps = 1e-8 
                )

epochs =10
skip_train = False
total_steps = len(bert_train_dataloader) * epochs

bert_scheduler = get_linear_schedule_with_warmup(bert_optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
seed_val = 100

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

bert_training_stats = []

total_t0 = time.time()

if not skip_train:
    for epoch_i in range(0, epochs):
        # ========================================
        #               Training
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        total_train_loss = 0
        bert_model.train()
        for step, batch in enumerate(bert_train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(bert_train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            bert_model.zero_grad()
            outputs = bert_model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
            bert_optimizer.step()
            bert_scheduler.step()
        avg_train_loss = total_train_loss / len(bert_train_dataloader)
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

if skip_train:  
    bert_model = BertForSequenceClassification.from_pretrained('bert_save_dir/')
else: 
    bert_model.save_pretrained('bert_save_dir/')
    with open('bert_training_stats.txt', 'w') as filehandle:
        json.dump(bert_training_stats, filehandle)
# ========================================
#               Testing
# ========================================
print('Number of test sentences: {:,}\n'.format(df_test.shape[0]))
sentences = df_test['clean_title'].values
labels = df_test['label'].values
input_ids = []
attention_masks = []

for sent in sentences:
    encoded_dict = bert_tokenizer.encode_plus(
                        sent,                      
                        add_special_tokens = True, 
                        max_length = 512,           
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',    
                   ) 
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels) 
batch_size = 16

prediction_data = TensorDataset(input_ids, attention_masks, labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))
bert_model.eval()
predictions , true_labels = [], []
total_eval_accuracy=0

for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        outputs = bert_model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)  
    logits = outputs[0]  
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    predictions.append(logits)
    true_labels.append(label_ids)
    total_eval_accuracy += flat_accuracy(logits, label_ids)

avg_val_accuracy = total_eval_accuracy / len(prediction_dataloader)
print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
print('    DONE.')
print('Positive samples: %d of %d (%.2f%%)' % (df_test['label'].sum(), len(df_test['label']), (df_test['label'].sum() / len(df_test['label']) * 100.0)))

flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

df_test['{}_way_pred'.format(num_of_way)] = flat_predictions
flat_true_labels = np.concatenate(true_labels, axis=0)

def get_eval_report(labels, preds):
    mcc = matthews_corrcoef(labels, preds)
    acc=accuracy_score(labels, preds)
    f1=f1_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    pre=tp/(tp+fp)
    rec=tp/(tp+fn)
    return {
        "acc": acc,
        "pre":pre,
        "rec":rec,
        "f1": f1,
    }
eval_report = get_eval_report(flat_true_labels, flat_predictions)
print("eval summary: ", eval_report)

with open('gossipcop_eval_report.json', 'w') as filehandle2:
    json.dump(eval_report, filehandle2, cls=NumpyEncoder)

def get_eval_report(task_name, labels, preds):
    mcc = matthews_corrcoef(labels, preds)
    acc= accuracy_score(labels,preds)
    f1=f1_score(labels,preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {
        "task": task_name,
        "acc":acc,
        "f1":f1,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }