import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

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

def insert_punctuation_marks(words):
  PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
  PUNC_RATIO = 0.3

  # words = sentence.split(' ')
  new_line = []

  q = random.randint(1, int(PUNC_RATIO * len(words) + 1))
  qs = random.sample(range(0, len(words)), q)

  for j, word in enumerate(words):
    if j in qs:
      new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS) -1)])
      new_line.append(word)
    else:
      new_line.append(word)
  new_line = ' '.join(new_line)
  return new_line
  
def customAeda(dataset, tokenizer):
  # aeda
  NUM_AUGS = [1, 2, 4, 8] # augmantation 횟수
  labels = []
  concat_entity = []
  sentences = []
  no_relation_count = Counter(list(dataset['label']))['no_relation'] # no_relation 데이터 수
  except_labels = set('no_relation')

  for aug in NUM_AUGS:
    for subj, obj, sentence, label in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence'], dataset['label']):
      # 기존 문장 추가
      temp = ''
      temp = subj + '[SEP]' + obj
      if aug == 1:
        concat_entity.append(temp)
        sentences.append(sentence)
        labels.append(label)

      # aeda # no_relation 제외 
      '''
      no_relation의 데이터 수가 가장 많아 no_relation을 기준으로 다른 label의 데이터 수를 비교하여
      augmentation을 진행함
      '''
      if label != 'no_relation':
        max_count = no_relation_count # augmantation 기준
        for i in range(aug):
          if label in list(except_labels): # 데이터가 except_labels에 포함되어있는 label에 해당하는 경우 종료
            break
          
          # 문장을 ' ' 기준으로 미리 나눈 후 entity에 해당하는 단어가 온전한지 확인
          # entity 에 ' '가 포함되어 있는 경우에는 split()에 의해 분리되기 때문에
          # 아래 과정을 통해 다시 합침
          words = sentence.split(' ')
          word_idx = 0
          while word_idx < len(words)-1:
            if ' ' not in subj and ' ' not in obj:
              break
            if (words[word_idx] in subj and words[word_idx+1] in subj):
              words[word_idx] = ' '.join(words[word_idx:word_idx+2])
              words.pop(word_idx+1)
            elif (words[word_idx] in subj and subj not in words[word_idx:word_idx+2] and subj in ' '.join(words[word_idx:word_idx+2])):
              words[word_idx] = ' '.join(words[word_idx:word_idx+2])
              words.pop(word_idx+1)
            if (words[word_idx] in obj and words[word_idx+1] in obj):
              words[word_idx] = ' '.join(words[word_idx:word_idx+2])
              words.pop(word_idx+1)
            elif (words[word_idx] in obj and obj not in words[word_idx:word_idx+2] and obj in ' '.join(words[word_idx:word_idx+2])):
              words[word_idx] = ' '.join(words[word_idx:word_idx+2])
              words.pop(word_idx+1)
            else:
              word_idx += 1


          concat_entity.append(temp)
          labels.append(label)
          sentence_aug = insert_punctuation_marks(words) # sentence에 puhctuation marks를 넣음
          sentences.append(sentence_aug) 
        label_counts = Counter(labels).most_common()
        max_count *= 0.4 # augmantation 기준

        for count in label_counts:
          if count[1] > max_count: # max_count 보다 많은 데이터를 가진 label에 해당하면 augmantation 하지 않음
            except_labels.add(count[0])

          
  tokenized_sentences = tokenizer(
    concat_entity,
    sentences,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
    add_special_tokens=True,
    return_token_type_ids=False,
  )

  return labels, tokenized_sentences

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


