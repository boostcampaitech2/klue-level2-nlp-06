# -*- coding: utf-8 -*-
"""Exp02_DomainAdaptation.ipynb

# Domain Adaption
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


"""## 1. Data Load & Preprocessing"""

train_dir = '../../dataset/train/train.csv'
test_dir = '../../dataset/test/test_data.csv'

pd_dataset = pd.read_csv(train_dir)
pd_dataset['sentence'].to_csv('data_sentence.txt', index=False, header=False, encoding='utf-8')


"""## 2. Load Model & Tokenizer"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
MODEL_NAME = 'klue/roberta-large'

# Domain-pre-training corpora
dpt_corpus_train = 'data_sentence.txt'
dpt_corpus_train_data_selected = 'data_sentence_selected.txt'
dpt_corpus_val = 'data_sentence_val.txt'

# Fine-tuning corpora
# If there are multiple downstream NLP tasks/corpora, you can concatenate those files together
ft_corpus_train = 'vocab.txt'

# Load Model & Tokenizer
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


"""## 3. Data Selection"""

from pathlib import Path
from transformers_domain_adaptation import DataSelector

selector = DataSelector(
    keep=0.5,  # TODO Replace with `keep`
    tokenizer=tokenizer,
    similarity_metrics=['euclidean'],
    diversity_metrics=[
        "type_token_ratio",
        "entropy",
    ],
)

# Load text data into memory
fine_tuning_texts = Path(ft_corpus_train).read_text(encoding='utf-8').splitlines()
training_texts = Path(dpt_corpus_train).read_text(encoding='utf-8').splitlines()

# Fit on fine-tuning corpus
selector.fit(fine_tuning_texts)

# Select relevant documents from in-domain training corpus
selected_corpus = selector.transform(training_texts)

# Save selected corpus to disk under `dpt_corpus_train_data_selected`
Path(dpt_corpus_train_data_selected).write_text('\n'.join(selected_corpus), encoding='utf-8');

selected_corpus[0]


"""## 4. Vocabulary Augmentation"""

from transformers_domain_adaptation import VocabAugmentor

target_vocab_size = 32500  # len(tokenizer) == 30_522

augmentor = VocabAugmentor(
    tokenizer=tokenizer, 
    cased=False,
    target_vocab_size=target_vocab_size
)

# Obtain new domain-specific terminology based on the fine-tuning corpus
#new_tokens = augmentor.get_new_tokens(ft_corpus_train)
new_tokens = augmentor.get_new_tokens(open(dpt_corpus_train, 'rt', encoding='UTF8'))

new_tokens

# Update model and tokenizer with new vocab terminologies
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))


"""## 5. Domain Pre-Training"""

import itertools as it
from pathlib import Path
from typing import Sequence, Union, Generator

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

datasets = load_dataset(
    'text', 
    data_files={
        "train": dpt_corpus_train, 
        "val": dpt_corpus_train_data_selected
    }
)

tokenized_datasets = datasets.map(
    lambda examples: tokenizer(examples['text'], truncation=True, max_length=model.config.max_position_embeddings), 
    batched=True
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./results/domain_pre_training",
    overwrite_output_dir=True,
    max_steps=100,
    per_device_train_batch_size=40,
    per_device_eval_batch_size=40,
    evaluation_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    logging_steps=50,
    seed=42,
    fp16=True,
    dataloader_num_workers=2,
    disable_tqdm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['val'],
    data_collator=data_collator,
    tokenizer=tokenizer,  # This tokenizer has new tokens
)

trainer.train()