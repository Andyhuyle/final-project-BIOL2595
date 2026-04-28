import pandas as pd

panda = pd.read_csv("/oscar/data/shared/ursa/kaggle_panda/train.csv")  # PANDA train labels
print(panda["isup_grade"].value_counts().sort_index())
print(panda["gleason_score"].value_counts())