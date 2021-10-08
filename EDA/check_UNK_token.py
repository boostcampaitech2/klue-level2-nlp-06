import pandas as pd
import numpy as np

from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer, XLMRobertaTokenizerFast

def preprocessing_dataset(dataset):
  """ csv 파일을 DataFrame으로 변경"""
  subject_entity = []
  object_entity = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = eval(i)['word'] # eval(): str -> dict
    j = eval(j)['word']

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset


dataset = pd.read_csv('/opt/ml/dataset/train/train.csv')
pre_dataset = preprocessing_dataset(dataset)

tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base") # 여기에 모델 넣고 실험하시면 됩니다.
sentence = pre_dataset['sentence'].tolist()

result = []
for idx in range(len(sentence)):
    result.append(tokenizer(
        sentence[idx],
        return_tensors="pt"
    ))

encoded_sentences = []
for idx in range(len(result)):
    sentence = [tokenizer.convert_ids_to_tokens(sentence) for sentence in result[idx]['input_ids']]
    encoded_sentences.append(sentence[0])

# 어떤 형태로 tokenizing되는지 보고 싶으면 키시면 됩니다.
# encoded_sentences[0] 

unk = 0
unk_list = []

for encoded_sentence in encoded_sentences:
    unk = 0
    for token in encoded_sentence:
        if token == '[UNK]':
            unk += 1
    unk_list.append(unk)

pre_dataset['UNK'] = unk_list

# token의 data 내의 문장에 UNK가 몇개 있는지 확인할 수 있습니다.
print(pre_dataset['UNK'].value_counts().sort_values())

