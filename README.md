<h1 align="center">
BoostCamp AI Tech - [NLP] 문장 내 개체간 관계 추출
</h1>
<p align="center">
    <a href="https://boostcamp.connect.or.kr/program_ai.html">
        <img src="https://img.shields.io/badge/BoostCamp-P--Stage-bronze?logo=Naver&logoColor=white"/></a> &nbsp
    </a>
    <a href=https://github.com/KLUE-benchmark/KLUE">
        <img src="https://img.shields.io/badge/Dataset-KLUE--RE-critical?logo=GitHub&logoColor=white"/></a> &nbsp
    </a>
    <a href="https://huggingface.co/klue/roberta-large">
        <img src="https://img.shields.io/badge/KLUE-roberta--large-yellow"/></a> &nbsp
    </a>
    <a href="https://en.wikipedia.org/wiki/F-score">
        <img src="https://img.shields.io/badge/Score (Micro--F1)-72.324-bronze"/></a> &nbsp
    </a>
</p>

<h3 align="center">
<p>Relation Extraction For Korean
</h3>

RE is a task to identify semantic relations between entity pairs in a text. The relation is defined between an entity pair consisting of subject entity and object entity. The goal is then to pick an appropriate relationship between these two entities in Korean sentence.
## Project Overview 
#### 프로젝트 목표
- 주어진 문장에서 Subject Entity와 Object Entity 사이의 관계를 예측하는 프로젝트
  
#### 데이터셋
- KLUE Dataset의 RE Task데이터로 30개의 관계가 존재하고, 약 32000개의 문장을 학습데이터로 학습한다.
- No relation(관계없음)의 분포가 x개로 가장 많고 (최소 라벨)의 분포가 x개로 가장 적었다.
  
#### 데이터 전처리 
- 동일한 문장, entity임에도 라벨이 달랐던 데이터가 존재했고 둘 중 올바른 라벨로 수정하였다.
  
#### 평가지표 
- 'No relation' 라벨을 제외한 Micro F1-score로 평가하였다. 
  

## Table of Contents
1. [Prerequisites Installatioin](#prerequisites-installatioin)
2. [Quick Start](#quick-start)
3. [Advanced Examples](#advanced-examples)
4. [Code Structure](#code-structure)
5. [Usage](#usage)
6. [Augmenters](#augmenters)
7. [Contributor](#contributor)
    
  
## Prerequisites Installatioin
requirements.txt can be installed using pip as follows:
```shell script
$ pip install -r requirements.txt
```

## Quick Start
- Train
```shell script
python train.py
```
- inference
```shell script
python inference.py
```   
 
## Advanced Examples
change the mode by editing the config.json
## Usage
### Using Focal loss

```json
    "focal_loss":{
        "true" : True,
        "alpha" : 0.1,
        "gamma" : 0.25
      },
```
### Using Imbalanced Sampler
```json
"Trainer" : {
      "use_imbalanced_sampler" : true 
    },
```

### Using Tokenize like BERT
**BERT result**
![](https://i.imgur.com/0vtBNj9.png)
[CLS] the man went to [MASK] store <span style="color:red">[SEP]</span>he bought a gallon [MASK] milk <span style="color:red">[SEP]</span> LABEL = IsNext
**like BERT result**
[CLS][obj] 변정수[/obj] 씨는 1994년 21살의 나이에 7살 연상 남편과 결혼해 슬하에 두 딸 [subj]유채원[/subj], 유정원 씨를 두고 있다. <span style="color:red">[SEP]</span>[obj][PER]변정수[/obj][subj][PER]유채원[/subj] <span style="color:red">[SEP]</span>
```json
"dataPP" :{ 
    "active" : true,
    "entityInfo" : "entity&token",
    "sentence" : "entity"
},
```
    
### AEDA 

```json
"aeda" : "None"
```
**Default**
하위 15개 label에 대해 AEDA 적용 (Mecab 설치필요)
```json
"aeda" : "default"
```
#### Mecab 설치방법
    sudo apt install g++
    sudo apt update
    sudo apt install default-jre
    sudo apt install default-jdk
    pip install konlpy

    # install khaiii
    cd ~
    git clone https://github.com/kakao/khaiii.git
    cd khaiii
    mkdir build
    cd build
    pip install cmake
    sudo apt-get install cmake
    cmake ..
    make resource
    sudo make install
    make package_python
    cd package_python
    pip install .
    cd ~
    apt-get install locales
    locale-gen en_US.UTF-8
    pip install tweepy==3.7.0
    # install mecab
    wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
    tar xvfz mecab-0.996-ko-0.9.2.tar.gz
    cd mecab-0.996-ko-0.9.2
    ./configure
    make
    make check
    sudo make install
    sudo ldconfig
    cd ~
    wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
    tar xvfz mecab-ko-dic-2.1.1-20180720.tar.gz
    cd mecab-ko-dic-2.1.1-20180720
    ./configure
    make
    sudo make install
    cd ~
    mecab -d /usr/local/lib/mecab/dic/mecab-ko-dic
    apt install curl
    apt install git
    bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
    pip install mecab-python

**Custom** 
Mecab 설치 불필요

`# of no_relation * 0.4` 보다 적은 데이터를 가지는 label에 대해서 augmantation 실행 
sentence를 space(' ')기준으로 나눈 후 entity에 해당하는 데이터를 합친 후 aeda 적용

```json
"aeda" : "custom"
    

## Config Augmenters
### Wandb
- RoRERTa-large

| Argument        | DataType    | Default                                  | Help                          |
|-----------------|:-----------:|:----------------------------------------:|:-----------------------------:|
| name            | str         | "roberta_large_stratified"                | Wandb model Name              |
| tags            | list        | ["ROBERT_LARGE", "stratified", "10epoch"]| Wandb Tags     |
| group           | str         | "ROBERT_LARGE"                            | Wandb group Name     |

- XLM-RoBERTa-large

| Argument        | DataType    | Default                                  | Help                          |
|-----------------|:-----------:|:----------------------------------------:|:-----------------------------:|
| name            | str         | "XLM-RoBERTa-large"                | Wandb model Name              |
| tags            | list        | ["XLM-RoBERTa-large", "stratified", "10epoch"]| Wandb Tags     |
| group           | str         | "XLM-RoBERTa-large"                            | Wandb group Name     |

### Focal Loss
| Argument        | DataType    | Default                                  | Help                          |
|-----------------|:-----------:|:----------------------------------------:|:-----------------------------:|
| true            | bool        | false                                     | Using Focal loss              |
| alpha           | float       | 0.1                                       | balances focal loss     |
| gamma           | float       | 0.25                                      | smoothly adjusts the rate  |

### Train Arguments

- RoBERTa-large

| Argument        | DataType    | Default          | Help                          |
|-----------------|:-----------:|:----------------:|:-----------------------------:|
| output_dir      | str         | "./results"      | result director                |
| save_total_limit| int         | 10               | limit of save files     |
| save_steps      | int         | 100              | saving step    |
| num_train_epochs| int         | 3                | train epochs |
| learning_rate   | int         | 5e-5             | learning rate |
| per_device_train_batch_size| int         | 38               | train batch size |
| per_device_eval_batch_size | int         | 38  | evaluation batch size        |
| warmup_steps    | int         | 500              | lr scheduler warm up step      |
| weight_decay    | float         | 0.01         | AdamW weight decay  |
| logging_dir      | str       | "./logs"             | logging dir   |
| logging_steps       | int         | 100            | logging step            |
| evaluation_strategy   | str         | "steps"               | evaluation strategy (epoch or step) |
| eval_steps    | int     | 100               | eval steps |
| load_best_model_at_end   | bool         | true     |  best checkpoint saving (loss) |

- XLM-RoBERTa-large

| Argument        | DataType    | Default          | Help                          |
|-----------------|:-----------:|:----------------:|:-----------------------------:|
| output_dir      | str         | "./results"      | result director                |
| save_total_limit| int         | 10               | limit of save files     |
| save_steps      | int         | 100              | saving step    |
| num_train_epochs| int         | 10                | train epochs |
| learning_rate   | int         | 5e-5             | learning rate |
| per_device_train_batch_size| int         | 31               | train batch size |
| per_device_eval_batch_size | int         | 31  | evaluation batch size        |
| warmup_steps    | int         | 500              | lr scheduler warm up step      |
| weight_decay    | float         | 0.01         | AdamW weight decay  |
| logging_dir      | str       | "./logs"             | logging dir   |
| logging_steps       | int         | 100            | logging step            |
| evaluation_strategy   | str         | "steps"               | evaluation strategy (epoch or step) |
| eval_steps    | int     | 100               | eval steps |
| load_best_model_at_end   | bool         | true     |  best checkpoint saving (loss) |


## Reference
[Easy Data Augmentation Paper](https://www.aclweb.org/anthology/D19-1670.pdf)  
[Korean WordNet](http://wordnet.kaist.ac.kr/)


# klue-level2-nlp-06
klue-level2-nlp-06 created by GitHub Classroom  
P stage  
KLUE  
Relation Extraction  


# Config.json Tutorial
* 왜 필요?
    * 실험할 때 train.py에서 다양한 설정들을 변경해야 함
        * num_epochs
        * models
        * ...
        * 이 상태로 commit & pull을 여러 사람이 하게 되면 각자 설정이 꼬임
    * 실험할 때 다양한 조건들을 config.json에서만 바꿔서 실행할 수 있음. 
* 사실 그렇게 안복잡해요. 한번만 바꾸면 편해요.

# Config 기준 파일 저장 위치
* 최고 성능 모델
   *  ./best_model/모델이름
* checkpoint 저장 위치
   * ./result/모델이름
* 모델이름은 confog.json에서 wandb의 name의 값으로 주는 값
* 이 값이 wandb 이름에도 적용 됨.
* 동일한 이름으로 실험할 때 exp뒤에 숫자 붙여서 업데이트

# num_hidden_layers
* 시도하시는 모델에 num_hidden_layer argument가 없으면 에러가 날 수도 있고, 에러가 안나고 그냥 변화 없이 돌아가는지 저는 알지 못해요!!
* 실험하실 때 huggingface docs나 실험하시는 모델 문서를 꼭 참고하시고 실험하시면 좋을 것 같습니다!
