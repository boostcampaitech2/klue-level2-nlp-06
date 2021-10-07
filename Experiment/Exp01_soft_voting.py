import numpy as np
import pandas as pd
import copy

label_dict = {0: 'no_relation', 1: 'org:top_members/employees', 2: 'org:members', 3: 'org:product', 4: 'per:title',
              5: 'org:alternate_names', 6: 'per:employee_of', 7: 'org:place_of_headquarters', 8: 'per:product',
              9: 'org:number_of_employees/members', 10: 'per:children', 11: 'per:place_of_residence', 12: 'per:alternate_names',
              13: 'per:other_family', 14: 'per:colleagues', 15: 'per:origin', 16: 'per:siblings', 17: 'per:spouse',
              18: 'org:founded', 19: 'org:political/religious_affiliation', 20: 'org:member_of', 21: 'per:parents',
              22: 'org:dissolved', 23: 'per:schools_attended', 24: 'per:date_of_death', 25: 'per:date_of_birth',
              26: 'per:place_of_birth', 27: 'per:place_of_death', 28: 'org:founded_by', 29: 'per:religion'}

model_1 = pd.read_csv("./ensemble/fix_error.csv")
model_2 = pd.read_csv("./ensemble/sep_next.csv")
model_3 = pd.read_csv("./ensemble/sota1_focal_loss.csv")
model_4 = pd.read_csv("./ensemble/sota2_hanjin.csv")
model_5 = pd.read_csv("./ensemble/total_aug.csv")

model_all = copy.deepcopy(model_1)
for i in range(len(model_1)):
    np_1 = np.array(eval(model_1['probs'].iloc[i]))
    np_2 = np.array(eval(model_2['probs'].iloc[i]))
    np_3 = np.array(eval(model_3['probs'].iloc[i]))
    np_4 = np.array(eval(model_4['probs'].iloc[i]))
    np_5 = np.array(eval(model_5['probs'].iloc[i]))

    np_all = (np_1 + np_2 + np_3 + np_4 + np_5) / 5
    pred = label_dict[np.argmax(np_all)]
    # print(np.argmax(np_1))
    # print(np.argmax(np_2))
    # print(np.argmax(np_all))
    # print(label_dict[np.argmax(np_all)])
    model_all['probs'].iloc[i] = str(np_all.tolist())
    model_all['pred_label'].iloc[i] = pred

model_all.to_csv("submission_ensemble.csv", index=False)