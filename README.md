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
5. [Augmenters](#augmenters)
6. [Contributor](#contributor)


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
- Baseline
```json
    "focal_loss":{
        "true" : false,
        "alpha" : 0.1,
        "gamma" : 0.25
      },
```
- Using Focal loss
```json
    "focal_loss":{
        "true" : True,
        "alpha" : 0.1,
        "gamma" : 0.25
      },
```


## Config Augmenters
### Wandb
| Argument        | DataType    | Default                                  | Help                          |
|-----------------|:-----------:|:----------------------------------------:|:-----------------------------:|
| name            | str         | "roberta_large_stratified"                | Wandb model Name              |
| tags            | list        | ["ROBERT_LARGE", "stratified", "10epoch"]| Wandb Tags     |
| group           | str         | "ROBERT_LARGE"                            | Wandb group Name     |

### Focal Loss
| Argument        | DataType    | Default                                  | Help                          |
|-----------------|:-----------:|:----------------------------------------:|:-----------------------------:|
| true            | bool        | false                                     | Using Focal loss              |
| alpha           | float       | 0.1                                       | balances focal loss     |
| gamma           | float       | 0.25                                      | smoothly adjusts the rate  |

### Train Augments
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
