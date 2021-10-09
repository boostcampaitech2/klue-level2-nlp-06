import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
import copy
import random
from koeda import AEDA

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
  object_entity = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
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

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
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
      max_length=256,
      add_special_tokens=True,
      return_token_type_ids=False,
      )
  return tokenized_sentences

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


def load_stratified_data_AEDA(dataset_dir):
    pd_dataset, df_valid = load_stratified_data(dataset_dir)

    # 하위 15개
    df_train = pd_dataset[(pd_dataset['label'] == "per:place_of_death") |
                                        (pd_dataset['label'] == "org:number_of_employees/members   ") |
                                        (pd_dataset['label'] == "org:dissolved") |
                                        (pd_dataset['label'] == "per:schools_attended   ") |
                                        (pd_dataset['label'] == "per:religion") |
                                        (pd_dataset['label'] == "org:political/religious_affiliation   ") |
                                        (pd_dataset['label'] == "per:siblings") |
                                        (pd_dataset['label'] == "per:product") |
                                        (pd_dataset['label'] == "org:founded_by") |
                                        (pd_dataset['label'] == "per:place_of_birth") |
                                        (pd_dataset['label'] == "per:other_family") |
                                        (pd_dataset['label'] == "per:place_of_residence") |
                                        (pd_dataset['label'] == "per:children") |
                                        (pd_dataset['label'] == "org:product") |
                                        (pd_dataset['label'] == "per:date_of_death")
    ]

    aeda = AEDA(
        morpheme_analyzer="Mecab", punc_ratio=0.3, punctuations=[".", ",", "!", "?", ";", ":"]
    )
    df_train_sen = copy.deepcopy(pd_dataset)
    for idx, sent in enumerate(df_train['sentence']):
        se = df_train.iloc[idx]['subject_entity']
        se = se.strip()[1:-1]
        ob = df_train.iloc[idx]['object_entity']
        ob = ob.strip()[1:-1]

        if len(list(map(str, sent.split(se)))) == 2 and len(list(map(str, sent.split(ob)))) == 2 :
            A, B = map(str, sent.split(se))
            if ob in A:
                sentA, sentB = map(str, A.split(ob))
                sentA = aeda(sentA)
                sentB = aeda(sentB)
                B = aeda(B)

                sentence = sentA + se + sentB + ob + B 
        
            elif ob in B:
                sentA, sentB = map(str, B.split(ob))
                sentA = aeda(sentA)
                sentB = aeda(sentB)
                A = aeda(A)
                sentence = A + se + sentA + ob + sentB
            if sentence != sent :
                new_data = {'id': idx, "sentence" : sentence, 'subject_entity': df_train.iloc[idx]['subject_entity'], 
                            'object_entity' : df_train.iloc[idx]['object_entity'], 'label' : df_train.iloc[idx]['label']}
                df_train_sen = df_train_sen.append(new_data, ignore_index= True)
                
    return df_train_sen, df_valid
