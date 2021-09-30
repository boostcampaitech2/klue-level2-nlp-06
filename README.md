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
