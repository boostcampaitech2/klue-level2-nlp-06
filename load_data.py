import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
import copy
import random
from koeda import AEDA
from collections import Counter

os.chdir('./KorEDA/')
from KorEDA.eda import *
os.chdir('../')

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
def add_special_token(tokenizer, special_token): #Tokenizer에 Special Token을 추가
  token_dict={
    "none" : [],
    "entity" : ['<obj>', '</obj>', '<subj>', '</subj>'],
    "type" : ['<PER>', '<ORG>', '<LOC>', '<DAT>', '<POH>', '<NOH>'],
    "entity&token" : ['<obj>', '</obj>', '<subj>', '</subj>', '</PER>', '</ORG>', '</LOC>', '</DAT>', '</POH>', '</NOH>'],
    "all" : ['<obj>', '</obj>', '<subj>', '</subj>', '</PER>', '</ORG>', '</LOC>', '</DAT>', '</POH>', '</NOH>', '<wikipedia>', '<wikitree>', '<policy_briefing>'], 
  }  
  entity_special_tokens = {'additional_special_tokens': token_dict[special_token]}    
  num_additional_special_tokens=  tokenizer.add_special_tokens(entity_special_tokens)    
  return tokenizer
def add_special_tokens_to_sentence(sentence, object_entity, subject_entity, source, special_token): #문장 내부에 Speical Token을 추가
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
def make_info_entity(object_entity, subject_entity, source, type): #Sentence[SEP]info_entity 의 info_entity를 생성
    obj_start_idx, obj_end_idx = object_entity['start_idx'], object_entity['end_idx']
    subj_start_idx, subj_end_idx = subject_entity['start_idx'], subject_entity['end_idx']  
    obj_type = type_tag_dict[object_entity['type']]
    subj_type = type_tag_dict[subject_entity['type']]    
    source_tag = '<'+source+'>'
    entity = ''
    '''     
    entity,type, entity&type, all
    '''
    obj_token = '<obj>'+object_entity['word'] +'</obj>'
    subj_token = '<subj>'+subject_entity['word'] +'</subj>'
    if type =='entity&type':
      if obj_start_idx < subj_start_idx:
        entity=obj_token+obj_type+subj_token+subj_type
      else :
        entity=subj_token+subj_type+obj_token+obj_type
    elif type=='entity':
      if obj_start_idx < subj_start_idx:
        entity=obj_token+subj_token
      else :
        entity=subj_token+obj_token
    elif type=='all' :
      if obj_start_idx < subj_start_idx:
        entity=source_tag+obj_token+subj_token
      else :
        entity=source_tag+subj_token+obj_token
    return entity
def read_klue_re(dataset, type):
    sentences = []
    entites = []
    labels = [] 
    for temp in dataset.iterrows():
        data=temp[1]
        sentence = add_special_tokens_to_sentence(data['sentence'], data['object_entity_dict'], data['subject_entity_dict'], data['source'], type['sentence'])
        entity = make_info_entity(data['object_entity_dict'], data['subject_entity_dict'], data['source'], type['entityInfo'])        
        sentences.append(sentence)
        entites.append(entity)
        labels.append(data['label'])
    return sentences, entites
class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item
  def get_labels(self):
    return self.labels

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

def tokenized_dataset(dataset, tokenizer, tok_len, type):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  if type['active'] : 
    sentences, entities = read_klue_re(dataset, type)
    tokenizer = add_special_token(tokenizer, type['entityInfo'])
    tokenized_sentences =tokenizer(
                        sentences,
                        entities,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=514,
                        add_special_tokens=True,
                        return_token_type_ids=False,
                        )  
  else : 
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
  return tokenized_sentences, len(tokenizer)  

def load_stratified_data(dataset_dir, aug_family = False, type_ent_marker = False, type_punct = False, aug_AEDA = False):
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
    strat_train_set = augment_family(strat_train_set)

  if type_ent_marker:
    strat_train_set['sentence']  = entity_marker(strat_train_set, typed_punct = type_punct)
    strat_dev_set['sentence'] = entity_marker(strat_dev_set, typed_punct = type_punct)

  train_dataset = preprocessing_dataset(strat_train_set)  
  dev_dataset = preprocessing_dataset(strat_dev_set)  

  if aug_AEDA :
    train_dataset, dev_dataset = AEDA_augmentation(train_dataset, dev_dataset)
    
  return train_dataset, dev_dataset

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
        max_count *= 0.6 # augmantation 기준

        for count in label_counts:
          if count[1] > max_count: # max_count 보다 많은 데이터를 가진 label에 해당하면 augmantation 하지 않음
            except_labels.add(count[0])

          
  tokenized_sentences = tokenizer(
      concat_entity,
      sentences,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      return_token_type_ids=False,
      )
  return labels, tokenized_sentences


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

def augment_family(dataset):
    """
    Function: augmentation
    Definition: per:children", "per:colleagues", "per:other_family 레이블에 대해서 augmentation. KorEDA의 randomSwap 사용
    """
    before = len(dataset)
    print("processing data augmentation... may take a while...")

    for label in [ "per:children", "per:colleagues", "per:other_family" ]:
        target_set = dataset[dataset.label == label]
        new_example = augmentation(target_set)
        new_df = pd.DataFrame(new_example, columns = target_set.columns)
        new_df
        dataset = pd.concat([dataset, new_df])

    after = len(dataset)    
    print("augmentation finishied. Before aug size: %d, After aug size: %d" %(before, after))
    return dataset


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

# AEDA를 사용하기 위한 data load 함수
def AEDA_augmentation(pd_dataset, df_valid):
    # label 개수 하위 15개만 따로 빼서 Augmentation을 진행하기 위해 data 추출
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
    # entity안에 AEDA 적용이 들어가지 않도록 하기 위한 코드 
    for idx, sent in enumerate(df_train['sentence']):
        se = df_train.iloc[idx]['subject_entity']
        se = se.strip()[1:-1]
        ob = df_train.iloc[idx]['object_entity']
        ob = ob.strip()[1:-1]

        if len(list(map(str, sent.split(se)))) == 2 and len(list(map(str, sent.split(ob)))) == 2 :
            A, B = map(str, sent.split(se))
            if ob in A:
                sentA, sentB = map(str, A.split(ob))
                if len(B) > 1 and len(sentA) > 1 and len(sentB) > 1:
                    sentA = aeda(sentA)
                    sentB = aeda(sentB)
                    B = aeda(B)
                sentence = sentA + se + sentB + ob + B 
        
            elif ob in B:
                sentA, sentB = map(str, B.split(ob))
                if len(A) > 1 and len(sentA) > 1 and len(sentB) > 1:
                    sentA = aeda(sentA)
                    sentB = aeda(sentB)
                    A = aeda(A)
                sentence = A + se + sentA + ob + sentB
            # 문장이 맞으면 train set에 생성된 data 추가
            if sentence != sent :
                new_data = {'id': idx, "sentence" : sentence, 'subject_entity': df_train.iloc[idx]['subject_entity'], 
                            'object_entity' : df_train.iloc[idx]['object_entity'], 'label' : df_train.iloc[idx]['label']}
                df_train_sen = df_train_sen.append(new_data, ignore_index= True)
                
    return df_train_sen, df_valid

