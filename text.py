import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import os
from StemmedCountVectorizer import StemmedCountVectorizer

vectorizer = StemmedCountVectorizer(min_df=1, stop_words='english')


'''content = ["How to format my hard disk", " Hard disk format problems "]
X=vectorizer.fit_transform(content)
print( vectorizer.get_feature_names())
print(X.toarray().transpose())'''

posts = [open(os.path.join("DIR", f)).read() for f in os.listdir("DIR")]
X_train=vectorizer.fit_transform(posts)
print(vectorizer.get_feature_names())


new_post = ["imaging databases"]
new_post_vec = vectorizer.transform(new_post)

def tfidf(term, doc, docset):
    '''kalameh dar matn / kalameh dar kole motun'''
    tf = float(doc.count(term)) / sum(doc.count(w) for w in docset)
    idf = sp.math.log(float(len(docset)) / (len([doc for doc in docset if term in doc])))
    return tf * idf

def dist_raw(v1, v2):
    v1n= v1/sp.linalg.norm(v1.toarray())
    v2n = v2 / sp.linalg.norm(v2.toarray())
    delta = v1n - v2n
    return sp.linalg.norm(delta.toarray())

best_dis=100
best_i=0
for i in range(5):
    dis=dist_raw(new_post_vec,X_train.getrow(i))
    print("post %i distance is %.4f" %(i,dis))
    if(dis<best_dis):
        best_dis=dis
        best_i=i

print("best mactch is index %i with dis=%.4f" %(best_i, best_dis))

print(X_train.toarray())