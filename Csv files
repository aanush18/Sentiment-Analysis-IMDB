train_path="Documents/aclImdb/train/"
test_path = "Documents/aclImdb/test/" # test data for grade evaluation. 

'''
IMDB_DATA_PREPROCESS explores the neg and pos folders from aclImdb/train and creates a output_file in the required format
Inpath - Path of the training samples 
Outpath - Path were the file has to be saved 
Name  - Name with which the file has to be saved 
Mix - Used for shuffling the data 
'''
def imdb_data_preprocess(inpath, outpath="./", name="imdb_te.csv", mix=False):
	import pandas as pd 
	from pandas import DataFrame, read_csv
	import os
	import csv 
	import numpy as np 

	stopwords = open("stopwords.en.txt", 'r' , encoding="ISO-8859-1").read()
	stopwords = stopwords.split("\n")

	indices = []
	text = []
	rating = []

	i =  0 

	for filename in os.listdir(inpath+"pos"):
		data = open(inpath+"pos/"+filename, 'r' , encoding="ISO-8859-1").read()
		data = remove_stopwords(data, stopwords)
		indices.append(i)
		text.append(data)
		rating.append("1")
		i = i + 1

	for filename in os.listdir(inpath+"neg"):
		data = open(inpath+"neg/"+filename, 'r' , encoding="ISO-8859-1").read()
		data = remove_stopwords(data, stopwords)
		indices.append(i)
		text.append(data)
		rating.append("0")
		i = i + 1

	Dataset = list(zip(indices,text,rating))
	
	if mix:
		np.random.shuffle(Dataset)

	df = pd.DataFrame(data = Dataset, columns=['row_Number', 'text', 'polarity'])
	df.to_csv(outpath+name, index=False, header=True)

	pass
  if __name__ == "__main__":
	import time
	start = time.time()
	print ("Preprocessing the training_data--")
	imdb_data_preprocess(inpath=train_path, mix=True)
	print ("Done with preprocessing. Now, will retreieve the training data in the required format")
	
	[Xtrain_text,Ytrain] = retrieve_data()
  imdb_data_preprocess(inpath=test_path, mix=True)
  Xtest_text=retrieve_data()
	
	
