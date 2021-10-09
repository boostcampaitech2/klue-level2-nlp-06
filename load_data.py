import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

os.chdir('./KorEDA/')
from KorEDA.eda import *
os.chdir('../')

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

def tokenized_dataset(dataset, tokenizer, tok_len):
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
      max_length=tok_len,
      add_special_tokens=True,
      return_token_type_ids=False,
      )
  return tokenized_sentences


def augmentation(target_set):
  """
  Function: augmentation
  Definition: per:children", "per:colleagues", "per:other_family 레이블에 대해서 augmentation. KorEDA의 randomSwap 사용
  """
  new_row = []
  for idx, sentence in enumerate(target_set.sentence):
    words = sentence.split(' ')
    words = [word for word in words if word is not ""]

    eg = target_set.iloc[idx]

    sub = eval(eg['subject_entity'])
    sub_ent = sub['word']
    sub_len = sub['end_idx'] - sub['start_idx']


    obj = eval(eg['object_entity'])
    obj_ent = obj['word']
    obj_len = obj['end_idx'] - obj['start_idx']

    times = 0
    
    trial = 0
    num_failed = 0
    while True:
      res = random_swap(words, 5)
      res = " ".join(res)
      trial += 1
      if trial == 10:
        num_failed += 1
        break

      if res.find(sub_ent) == -1 or res.find(obj_ent) == -1:
        continue

      _sub = sub.copy()
      _sub['start_idx'] = res.index(sub_ent)
      _sub['end_idx'] = _sub['start_idx'] + sub_len

      _obj = obj.copy()
      _obj['start_idx'] = res.index(obj_ent)
      _obj['end_idx'] = _obj['start_idx'] + obj_len

      new_row.append( [ len(target_set) + times + 1, res, str(_sub), str(_obj), eg['label'], eg['source'] ] )
      times += 1
      
      if times == 10:
        break
  return new_row


def add_ent_marker(sen, ent, sub_obj_type, ent_s, ent_e, add_len = 0, typed_punct = False):
  """
    Function: add_ent_marker
    Definition: entity_marker을 실제로 수행. 입력 문장에 <S:PER> 등의 typed entity을 추가, type_ent_marker을 적용하되 @ * * @ 등의 punctuation을 적용
    Reference: An Improved Baseline for Sentence-level Relation Extraction, Zhowu et al. (2017)
  """

  ent_type = ent['type']
  if add_len > 0:
    ent_s += add_len
    ent_e += add_len
  
  front = sen[:ent_s]
  tail = sen[ent_e+1:]

  if typed_punct: # typed entitiy with punctuation
    if sub_obj_type == "S":
      s_tok = "@ * " + ent_type  + " * "
      e_tok = " @"
    else:
      s_tok = "# ^ " + ent_type  + " ^ "
      e_tok = " #"

  else: # type entity
    s_tok = "<" + sub_obj_type + ":" + ent_type + ">"
    e_tok = "</"+ sub_obj_type + ":" + ent_type + ">"

  word = s_tok + ent['word'] + e_tok

  res = front + word + tail

  add_len = len(res) - len(sen)
  return res, add_len

def entity_marker(dataset, typed_punct = False):
  """
    Function: entity_marker
    Definition: 입력 문장에 <S:PER> 등의 typed entity을 추가, type_ent_marker을 적용하되 @ * * @ 등의 punctuation을 적용
    Argument: 
      dataset: 적용할 데이터셋
      typed_punct: punctuation을 적용할 여부
    Reference: An Improved Baseline for Sentence-level Relation Extraction, Zhowu et al. (2017)
  """

  sentences = list(dataset['sentence'])
  subs = list(dataset['subject_entity'])
  objs = list(dataset['object_entity'])
  res_list = []
  for i, sen in enumerate(sentences):
    sub = eval(subs[i])
    sub_s = sub['start_idx']
    sub_e = sub['end_idx']

    obj = eval(objs[i])
    obj_s = obj['start_idx']
    obj_e = obj['end_idx']
    
    if obj_s < sub_s:
      res, add_len = add_ent_marker(sen, obj, "O", obj_s, obj_e, typed_punct = typed_punct)
      res, _ = add_ent_marker(res, sub, "S", sub_s, sub_e, add_len = add_len, typed_punct = typed_punct)
    
    else:
      res, add_len = add_ent_marker(sen, sub, "S", sub_s, sub_e, typed_punct = typed_punct)
      res, _ = add_ent_marker(res, obj, "O", obj_s, obj_e, add_len = add_len, typed_punct = typed_punct)
    res_list.append(res)
    # pd_dataset['res'][i] = res
  return res_list

def load_stratified_data(dataset_dir, aug_family = False, type_ent_marker = False, type_punct = False):
  """
  Function: load_stratified_data
  Definition: 데이터 비율에 맞춰서 validation을 분리
  Argument: 
    aug_family: per:children", "per:colleagues", "per:other_family 레이블에 대해서 augmentation
    type_ent_marker: 입력 문장에 <S:PER> 등의 typed entity을 추가
    type_punct: type_ent_marker을 적용하되 @ * * @ 등의 punctuation을 적용
      
  """
  """ csv 파일을 경로에 맡게 불러 옵니다. """  
  split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
  pd_dataset = pd.read_csv(dataset_dir)  
  
  for train_index, test_index in split.split(pd_dataset, pd_dataset["label"]):
    strat_train_set = pd_dataset.loc[train_index]
    strat_dev_set = pd_dataset.loc[test_index]

  if aug_family:
    before = len(strat_train_set)
    print("processing data augmentation... may take a while...")

    for l in [ "per:children", "per:colleagues", "per:other_family" ]:
        target_set = strat_train_set[strat_train_set.label == l]
        new_rows = augmentation(target_set)
        new_df = pd.DataFrame(new_rows, columns = target_set.columns)
        new_df
        strat_train_set = pd.concat([strat_train_set, new_df])

    after = len(strat_train_set)    
    print("augmentation finishied. Before aug size: %d, After aug size: %d" %(before, after))

  if type_ent_marker:
      res = entity_marker(strat_train_set, typed_punct = type_punct)
      strat_train_set['sentence'] = res

      res = entity_marker(strat_dev_set, typed_punct = type_punct)
      strat_dev_set['sentence'] = res

  train_dataset = preprocessing_dataset(strat_train_set)  
  dev_dataset = preprocessing_dataset(strat_dev_set)  
  return train_dataset, dev_dataset

