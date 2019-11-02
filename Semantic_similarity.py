import pandas as pd
X_train=pd.read_csv('imdb_tr.csv')
X_test=pd.read_csv('imdb_te.csv')
Y_train=X_train[X_train.columns[2:3]]
Y_test=X_test[X_test.columns[2:3]]
X_train=X_train[X_train.columns[0:2]]
X_test=X_test[X_test.columns[0:2]]
Xtrain_text=X_train[X_train.columns[1:2]]
def remove_stopwords(sentence, stopwords):
	sentencewords = sentence.split()
	resultwords  = [word for word in sentencewords if word.lower() not in stopwords]
	result = ' '.join(resultwords)
	return result


def unigram_process(data):
	from sklearn.feature_extraction.text import CountVectorizer
	vectorizer = CountVectorizer()
	vectorizer = vectorizer.fit(data)
	return vectorizer	


def bigram_process(data):
	from sklearn.feature_extraction.text import CountVectorizer
	vectorizer = CountVectorizer(ngram_range=(1,2))
	vectorizer = vectorizer.fit(data)
	return vectorizer


def tfidf_process(data):
	from sklearn.feature_extraction.text import TfidfTransformer 
	transformer = TfidfTransformer()
	transformer = transformer.fit(data)
	return transformer


def retrieve_data(name="imdb_tr.csv", train=True):
	import pandas as pd 
	data = pd.read_csv(name,header=0, encoding = 'ISO-8859-1')
	X = data['text']
	
	if train:
		Y = data['polarity']
		return X, Y

	return X		


def stochastic_descent(Xtrain, Ytrain, Xtest):
	from sklearn.linear_model import SGDClassifier 
	clf = SGDClassifier(loss="hinge", penalty="l1", n_iter=20)
	print ("SGD Fitting")
	clf.fit(Xtrain, Ytrain)
	print ("SGD Predicting")
	Ytest = clf.predict(Xtest)
	return Ytest


'''
ACCURACY finds the accuracy in percentage given the training and test labels 
Ytrain - One set of labels 
Ytest - Other set of labels 
'''
def accuracy(Ytrain, Ytest):
	assert (len(Ytrain)==len(Ytest))
	num =  sum([1 for i, word in enumerate(Ytrain) if Ytest[i]==word])
	n = len(Ytrain)  
	return (num*100)/n


'''
WRITE_TXT writes the given data to a text file 
Data - Data to be written to the text file 
Name - Name of the file 
'''
def write_txt(data, name):
	data = ''.join(str(word) for word in data)
	file = open(name, 'w')
	file.write(data)
	file.close()
	pass 


if __name__ == "__main__":
	import time
	start = time.time()

	uni_vectorizer = unigram_process(Xtrain_text)
	Xtrain_uni = uni_vectorizer.transform(Xtrain_text)

	bi_vectorizer = bigram_process(Xtrain_text)
	print ("Fitting the bigram model")
	Xtrain_bi = bi_vectorizer.transform(Xtrain_text)
	print ("After fitting ")
	
	uni_tfidf_transformer = tfidf_process(Xtrain_uni)
	print ("Fitting the tfidf for unigram model")
	Xtrain_tf_uni = uni_tfidf_transformer.transform(Xtrain_uni)
	print ("After fitting TFIDF")
	

	bi_tfidf_transformer = tfidf_process(Xtrain_bi)
	print ("Fitting the tfidf for bigram model")
	Xtrain_tf_bi = bi_tfidf_transformer.transform(Xtrain_bi)
	

	print ("-----------------------ANALYSIS ON THE TEST DATA ---------------------------")
	print ("Unigram Model on the Test Data--")
	Xtest_uni = uni_vectorizer.transform(Xtest_text)
	print ("Applying the stochastic descent")
	Ytest_uni = stochastic_descent(Xtrain_uni, Ytrain, Xtest_uni)
	write_txt(Ytest_uni, name="unigram.output.txt")
	

	print ("Bigram Model on the Test Data--")
	Xtest_bi = bi_vectorizer.transform(Xtest_text)
	Ytest_bi = stochastic_descent(Xtrain_bi, Ytrain, Xtest_bi)
	write_txt(Ytest_bi, name="bigram.output.txt")

	print ("Unigram TF Model on the Test Data--")
	Xtest_tf_uni = uni_tfidf_transformer.transform(Xtest_uni)
	Ytest_tf_uni = stochastic_descent(Xtrain_tf_uni, Ytrain, Xtest_tf_uni)
	write_txt(Ytest_tf_uni, name="unigramtfidf.output.txt")
	
	print ("Bigram TF Model on the Test Data--")
	Xtest_tf_bi = bi_tfidf_transformer.transform(Xtest_bi)
	Ytest_tf_bi = stochastic_descent(Xtrain_tf_bi, Ytrain, Xtest_tf_bi)
	write_txt(Ytest_tf_bi, name="bigramtfidf.output.txt")
	
	print ("Total time taken is ", time.time()-start, " seconds")
	pass
