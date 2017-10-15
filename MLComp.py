import sklearn.datasets
import matplotlib.pyplot as plt
from StemmedCountVectorizer import StemmedCountVectorizer
from sklearn.cluster import KMeans
import scipy as sp


mlcomp= r"MLCOMP"

groups = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.ma c.hardware', 'comp.windows.x', 'sci.space']
data = sklearn.datasets.load_mlcomp("20news-18828","train", mlcomp_root= mlcomp , categories=groups)
vectorizer=StemmedCountVectorizer(max_df=0.5, min_df=10, stop_words="english", decode_error='ignore')
vectorized = vectorizer.fit_transform(data.data)
num_samples, num_features = vectorized.shape
print("#samples: %d, #features: %d" % (num_samples, num_features))


new_post=["Disk drive problems. Hi, I have a problem with my hard disk. After 1 year it is working only sporadically now. I tried to format it, but now it doesn't boot any more. Any ideas? Thanks."]



num_clusters = 50
km = KMeans(n_clusters=num_clusters, init='random', n_init=1,verbose=1)
km.fit(vectorized)


a=sp.math.floor(num_features/2)

m=vectorized.toarray()

'''x=m[:,1:a]
y=m[:, a+1:-1]



plt.scatter(x, y ,color = 'red' )
plt.show()'''


new_post_vec = vectorizer.transform(new_post)
new_post_label = km.predict(new_post_vec)[0]
similar_indices = (km.labels_==new_post_label).nonzero()[0]
print(new_post_label)

similar = []
for i in similar_indices:
    dist = sp.linalg.norm((new_post_vec - vectorized[i]).toarray())
    similar.append((dist, data.data[i]))

similar = sorted(similar)

show_at_1 = similar[0]
show_at_2 = similar[sp.math.floor( len(similar) / 2)]
show_at_3 = similar[-1]

print(show_at_1)
print(show_at_2)
print(show_at_3)