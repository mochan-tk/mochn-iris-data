from sklearn import svm
from sklearn.metrics import f1_score
import pickle

train_x = [[0, 0], [1, 1]]
train_y = ['red', 'blue']
test_x = [[0, 0.5], [2, 2]]
test_y = ['red', 'blue']

model = svm.SVC()
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
print(f1_score(test_y, pred_y, labels=['red', 'blue'], pos_label='red'))

with open('model3.pkl', 'wb') as fout:
        pickle.dump(model, fout)

