import json

with open('config.json') as f:
    config = json.load(f)

# Instruction:

#   output_dir='./results',          # output directory
#   save_total_limit=5,              # number of total save model.
#   save_steps=500,                 # model saving step.
#   num_train_epochs=20,              # total number of training epochs
#   learning_rate=5e-5,               # learning_rate
#   per_device_train_batch_size=16,  # batch size per device during training
#   per_device_eval_batch_size=16,   # batch size for evaluation
#   warmup_steps=500,                # number of warmup steps for learning rate scheduler
#   weight_decay=0.01,               # strength of weight decay
#   logging_dir='./logs',            # directory for storing logs
#   logging_steps=100,              # log saving step.
#   evaluation_strategy='steps', # evaluation strategy to adopt during training
#                               # `no`: No evaluation during training.
#                               # `steps`: Evaluate every `eval_steps`.
#                               # `epoch`: Evaluate every end of epoch.
#   eval_steps = 500,            # evaluation step.
#   load_best_model_at_end = True 