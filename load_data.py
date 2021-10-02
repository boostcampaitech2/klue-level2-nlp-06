import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
type_tag_dict = {
      'PER' : '</PER>', 
      'ORG' : '</ORG>', 
      'LOC' : '</LOC>',
      'DAT' : '</DAT>',
      'POH' : '</POH>', 
      'NOH' : '</NOH>'
    }

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label
def add_special_token(tokenizer, special_token):
  token_dict={
    "none" : [],
    "entity" : ['<obj>', '</obj>', '<subj>', '</subj>'],
    "type" : ['<PER>', '<ORG>', '<LOC>', '<DAT>', '<POH>', '<NOH>'],
    "entity&type" : ['<obj>', '</obj>', '<subj>', '</subj>', '</PER>', '</ORG>', '</LOC>', '</DAT>', '</POH>', '</NOH>'],
    "all" : ['<obj>', '</obj>', '<subj>', '</subj>', '</PER>', '</ORG>', '</LOC>', '</DAT>', '</POH>', '</NOH>', '<wikipedia>', '<wikitree>', '<policy_briefing>'], 
  }  
  entity_special_tokens = {'additional_special_tokens': token_dict[special_token]}    
  num_additional_special_tokens=  tokenizer.add_special_tokens(entity_special_tokens)    
  return tokenizer
def add_special_tokens_to_sentence(sentence, object_entity, subject_entity, source, special_token):
    obj_start_idx, obj_end_idx = object_entity['start_idx'], object_entity['end_idx']
    subj_start_idx, subj_end_idx = subject_entity['start_idx'], subject_entity['end_idx']    
    obj_type = type_tag_dict[object_entity['type']]
    sbj_type = type_tag_dict[subject_entity['type']]    
    source_tag = '<'+source+'>'
    if special_token=='entity' :
      if obj_start_idx < subj_start_idx:
          new_sentence = sentence[:obj_start_idx] + '<obj>' + sentence[obj_start_idx:obj_end_idx+1] + '</obj>' + \
                        sentence[obj_end_idx+1:subj_start_idx] +'<subj>'+sentence[subj_start_idx:subj_end_idx+1] + \
                        '</subj>' + sentence[subj_end_idx+1:]
      else:
          new_sentence = sentence[:subj_start_idx] + '<subj>' + sentence[subj_start_idx:subj_end_idx+1] + '</subj>' + \
                        sentence[subj_end_idx+1:obj_start_idx] + '<obj>'+ sentence[obj_start_idx:obj_end_idx+1] + \
                        '</obj>' + sentence[obj_end_idx+1:]    
    elif special_token=='type' :
      if obj_start_idx < subj_start_idx:
          new_sentence = sentence[:obj_start_idx] + obj_type + sentence[obj_start_idx:obj_end_idx+1] + \
                        sentence[obj_end_idx+1:subj_start_idx] +sbj_type +sbj_type+sentence[subj_start_idx:subj_end_idx+1] + \
                        + sentence[subj_end_idx+1:]
      else:
          new_sentence = sentence[:subj_start_idx] + sbj_type + sentence[subj_start_idx:subj_end_idx+1] + \
                        sentence[subj_end_idx+1:obj_start_idx] + obj_type+ obj_type+ sentence[obj_start_idx:obj_end_idx+1] + \
                         + sentence[obj_end_idx+1:]   
    elif special_token=='entity&type' :
      if obj_start_idx < subj_start_idx:
          new_sentence = sentence[:obj_start_idx] + '<obj>'+obj_type + sentence[obj_start_idx:obj_end_idx+1] + '</obj>' + \
                        sentence[obj_end_idx+1:subj_start_idx] +'<subj>'+ sbj_type+sentence[subj_start_idx:subj_end_idx+1] + \
                        '</subj>' + sentence[subj_end_idx+1:]
      else:
          new_sentence = sentence[:subj_start_idx] + '<subj>'+sbj_type + sentence[subj_start_idx:subj_end_idx+1] + '</subj>' + \
                        sentence[subj_end_idx+1:obj_start_idx] + '<obj>'+obj_type+ sentence[obj_start_idx:obj_end_idx+1] + \
                        '</obj>' + sentence[obj_end_idx+1:] 
    elif special_token=='all' :
      if obj_start_idx < subj_start_idx:
          new_sentence = source_tag+sentence[:obj_start_idx] + '<obj>'+obj_type + sentence[obj_start_idx:obj_end_idx+1] + '</obj>' + \
                        sentence[obj_end_idx+1:subj_start_idx] +'<subj>'+ sbj_type+sentence[subj_start_idx:subj_end_idx+1] + \
                        '</subj>' + sentence[subj_end_idx+1:]
      else:
          new_sentence = source_tag+sentence[:subj_start_idx] + '<subj>'+sbj_type + sentence[subj_start_idx:subj_end_idx+1] + '</subj>' + \
                        sentence[subj_end_idx+1:obj_start_idx] + '<obj>'+obj_type+ sentence[obj_start_idx:obj_end_idx+1] + \
                        '</obj>' + sentence[obj_end_idx+1:] 
    else :
      new_sentence= sentence   
    return new_sentence
def read_klue_re(dataset, special_token):
    sentences = []
    labels = [] 
    for temp in dataset.iterrows():
        data=temp[1]
        sentence = add_special_tokens_to_sentence(data['sentence'], data['object_entity_dict'], data['subject_entity_dict'], data['source'], special_token)
        sentences.append(sentence)
        labels.append(data['label'])
    return sentences
class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  subject_entity_dict = []
  object_entity = []
  object_entity_dict = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):    
    subject_entity_dict.append(eval(i))    
    object_entity_dict.append(eval(i))
    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]    
    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],
                              'subject_entity':subject_entity,'object_entity':object_entity,
                              'label':dataset['label'], 'source' : dataset['source'],
                              'subject_entity_dict':subject_entity_dict,'object_entity_dict':object_entity_dict,
                              })
  duplied = out_dataset[out_dataset.duplicated(subset=['sentence','subject_entity','object_entity'])]
  duplied_no_idx = duplied[duplied['label'] == 'no_relation']['id'].to_list()
  for idx in duplied_no_idx:
    out_dataset.drop(out_dataset.loc[out_dataset['id']==idx].index, inplace=True)
  out_dataset = out_dataset.drop_duplicates(subset=['sentence','subject_entity','object_entity','label'],keep='first')  
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer, type='original'):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  if type == 'original' : 
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
      temp = ''
      temp = e01 + '[SEP]' + e02
      concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=514,
        add_special_tokens=True,
        return_token_type_ids=False,
        )    
  else : 
    sentences = read_klue_re(dataset, type)
    tokenizer = add_special_token(tokenizer, type)
    tokenized_sentences =tokenizer(
                        sentences,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=514,
                        add_special_tokens=True,
                        return_token_type_ids=False,
                        )  
  return tokenized_sentences, len(tokenizer)
  

def load_stratified_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """  
  split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
  pd_dataset = pd.read_csv(dataset_dir)  
  
  for train_index, test_index in split.split(pd_dataset, pd_dataset["label"]):
    strat_train_set = pd_dataset.loc[train_index]
    strat_dev_set = pd_dataset.loc[test_index]
  train_dataset = preprocessing_dataset(strat_train_set)  
  dev_dataset = preprocessing_dataset(strat_dev_set)  
  return train_dataset, dev_dataset


