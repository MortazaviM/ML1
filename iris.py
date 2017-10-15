
from matplotlib import pyplot as plt
import scipy as sp
import numpy as np
from sklearn.datasets import load_iris



data = load_iris()

features=data['data']
target=data['target']


for t,marker,c in zip(range(3),"<ox","rgb"):
# We plot each class on its own to get different colored markers
    plt.scatter(features[target == t,0],
    features[target == t,1],
    marker=marker, color=c)


plt.show()


plength = features[:, 2]
# use numpy operations to get setosa features
is_setosa = (target == 0)
# This is the important step:
max_setosa =plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()
print('Maximum of setosa: {0}.'.format(max_setosa))
print('Minimum of others: {0}.'.format(min_non_setosa))



best_acc = -1.0
for fi in range(features.shape[1]):
    # We are going to generate all possible threshold for this feature
    thresh = features[:,fi].copy()
    thresh.sort()
    # Now test all thresholds:
    for t in thresh:
        pred = (features[:,fi] > t)
        acc = (pred[target==1]).mean()
        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t


'''print(target)'''
print(best_t)
print(best_fi)

