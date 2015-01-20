
import sys
import jinja2
import sklearn
import json
import scipy
from scipy import sparse
import numpy
import sklearn.preprocessing
from sklearn.manifold import TSNE

#output file for d3 ---> could be snapshot
def get_row(dat_json):
	array = [ value for key,value in dat_json['dat'].iteritems()]
	return array

data_file = sys.argv[1]
X_vectors = []
attr_dicts = []
with open(data_file) as f:
	lines = f.readlines()
	for line in lines:
		dat = json.loads(line)
		row = get_row(dat)
		attr_dicts.append(dat['attr'])
		X_vectors.append(row)
X = scipy.vstack(X_vectors)

#normalize
X_scaled = sklearn.preprocessing.scale(X, axis=0)	
#T_SNE
model = TSNE(n_components=2, random_state=0)
TSNE_X = model.fit_transform(X)

out_data_name = data_file.split(".json")[0]+'.tsv'
with open(out_data_name,"w") as f:
	#write headers
	f.write("X\tY\t"+reduce((lambda x,y: x+'\t'+y),attr_dicts[0].keys())+"\n")
	#print dat
	for i in xrange(0,len(attr_dicts)):
		f.write(str(TSNE_X[i,0])+'\t'+str(TSNE_X[i,1])+'\t'+reduce(lambda x,y: str(x)+'\t'+str(y),attr_dicts[i].values())+'\n')
#read json

#generate matrix
#TSNE
#output .tsv file for d3
#do jinja templating on html



# X and Y properties
# generate print function
# generate view selected function
# how to handle colours?
# legend?




