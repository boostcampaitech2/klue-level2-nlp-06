import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, EarlyStoppingCallback
from load_data import *
import random
import wandb
from pathlib import Path
import glob
import re
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler

wandb.login()

from config_parser import config as cfg
class ImbalancedSamplerTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset

        def get_label(dataset):
            return dataset.get_labels()

        train_sampler = ImbalancedDatasetSampler(
            train_dataset, callback_get_label=get_label
        )

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
# code from mask competition
def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
                  'org:product', 'per:title', 'org:alternate_names',
                  'per:employee_of', 'org:place_of_headquarters', 'per:product',
                  'org:number_of_employees/members', 'per:children',
                  'per:place_of_residence', 'per:alternate_names',
                  'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
                  'per:spouse', 'org:founded', 'org:political/religious_affiliation',
                  'org:member_of', 'per:parents', 'org:dissolved',
                  'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
                  'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
                  'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    class_names = np.arange(30)

    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None, \
                            y_true=labels, preds=preds, \
                            class_names=class_names)})
                        

    return {
        'micro f1 score': f1,
        'auprc': auprc,
        'accuracy': acc,
    }


def label_to_num(label):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label

#entity, type, entity&type, all
def train(args):
    seed_everything(2021) # fix seed to current year

    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    MODEL_NAME = cfg['model']['huggingface']

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    

    # load dataset + prerprocessing dataset
    train_dataset, dev_dataset = load_stratified_data("../dataset/train/train.csv") 

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train, len_tokenizer = tokenized_dataset(train_dataset, tokenizer, cfg['dataPP']['specialToken'])
    tokenized_dev, _ = tokenized_dataset(dev_dataset, tokenizer,  cfg['dataPP']['specialToken'])

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    print(device)
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    
    for c, val in cfg['model']['config'].items():
        setattr(model_config, c, val)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config)
    model.resize_token_embeddings(len_tokenizer)
    print(model.config)    
    model.parameters
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(**args['training_arg'])

    # early stop argument
    callbacks_list = []
    if args['early_stop']:
        callbacks_list.append( EarlyStoppingCallback(early_stopping_patience = args['patience']) )
    if cfg['train']['Trainer']['use_imbalanced_sampler'] :   
        trainer = ImbalancedSamplerTrainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=RE_train_dataset,         # training dataset
            eval_dataset=RE_dev_dataset,             # evaluation dataset
            compute_metrics=compute_metrics,         # define metrics function
            callbacks = callbacks_list
        )
    else : 
        trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,         # define metrics function
        callbacks = callbacks_list
    )

    # train model
    trainer.train()
    model.save_pretrained('./best_model/' + args['exp_name'])
    wandb.finish()


def main():

    # append result output directory and rename with experiment number
    output_dir = cfg['train']['TrainingArguments']['output_dir']
    exp_name = cfg['wandb']['name']

    cfg['train']['TrainingArguments']['output_dir'] = increment_path(output_dir + "/" + exp_name + "_exp")        
    cfg['wandb']['name'] = cfg['train']['TrainingArguments']['output_dir'].split("/")[-1]
    print(cfg['wandb']['name'])
    print(cfg['train']['TrainingArguments']['output_dir'])

    args = {'training_arg' : cfg['train']['TrainingArguments'], \
            'exp_name' : exp_name,\
            'early_stop': cfg['train']['early_stop']['true']}

    #early stop
    if cfg['train']['early_stop']['true']:
        args['patience'] =  cfg['train']['early_stop']['patience']

    wandb.init(project='klue-RE', name=cfg['wandb']['name'],tags=cfg['wandb']['tags'], group=cfg['wandb']['group'], entity='boostcamp-nlp-06')
    
    train(args)


if __name__ == '__main__':
    main()
