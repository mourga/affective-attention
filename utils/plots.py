import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

f1 = [60, 61, 62]
acc = [71, 72, 73]

x = ['f1', 'accuracy', 'precision', 'recall']
models = ['baseline', 'baseline conc emb', 'baseline conc rnn', 'conc', 'gating',
          'affine', 'non linear affine']
f1 = [60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0]
accuracy = [60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0]
precision = [60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0]
recall = [60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0]

df = pd.DataFrame(
    {
     'models': models,
     'f1': f1,
     'accuracy': accuracy,
     'precision': precision,
     'recall': recall}
)
print()


# baseline = {'f1':60.0, 'accuracy':62.0, 'precision':61.0, 'recall':63.0}
# conc_emb = {'f1':58.0, 'accuracy':67.0, 'precision':62.0, 'recall':64.0}

baseline = [60.0, 61.0, 58.0, 62.0]
conc_emb = [62.0, 63.0, 59.0, 61.0]
metrics = ['f1', 'accuracy', 'precision', 'recall']

sentiment = [68.4, 68.4, 68.4, 68.6, 68.7, 68.5, 68.4]
irony = [46.3, 48.1, 46.6, 46.6, 47.1, 45.3, 46.6]
models = ['baseline', 'conc emb', 'conc rnn', 'conc', 'gating', 'affine', 'tanh affine']

fig, ax = plt.subplots()
sns.set(style='whitegrid')
ax.plot(models, sentiment, label="sentiment", marker='o', color='r')
ax.plot(models, irony, label="irony", marker='o', color='b')
ax.legend()
plt.savefig('metrics.png')

# plt.show()
# plt.figure()
# sns.set(style='whitegrid')
# g = sns.PairGrid(df)
#
# # titanic = sns.load_dataset("titanic")
# # sns.catplot(x="class", y="survived", hue="sex",
# #             palette={"male": "g", "female": "m"},
# #             markers=["^", "o"], linestyles=["-", "--"],
# #             kind="point", data=titanic)
# # plt.show()
