import csv
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.stem import WordNetLemmatizer 
import nltk

def read_data(filename):
	'Create a list of movie_id and the plots associated with each movie'

	df = pd.read_csv(filename)

	df = df[['id','overview']]
	df = df.dropna()

	plots = df['overview'].values

	tags_movies = df['id'].values

	return(plots, tags_movies)

def tokenize_plots(plots):

	'Tokenize each plot into words and lemmatize each word'
	
	plot_list_tokenzied = []
	lemmatizer = WordNetLemmatizer() 
	for i in range(len(plots)):
	    plots[i] = plots[i]
	    plot = word_tokenize(plots[i])
	    without_sw = []
	    for word in plot:
	        if word not in set(stopwords.words('english')) and word.isalpha():
	            word = lemmatizer.lemmatize(word.lower())
	            without_sw.append(word)
	    plot_list_tokenzied.append(without_sw)

	return(plot_list_tokenzied)



def doc_2_vec_model(plot_list_tokenzied, tags_movies):

	'Train a doc2vec model to get word embeddings for each movie plot'

	tagged_data = [TaggedDocument(doc, str(i)) for i, doc in enumerate(plot_list_tokenzied)]
	max_epochs = 100
	alpha = 0.025

	model = Doc2Vec(alpha=alpha,
					size=300,
	                workers = 6, #Number of cores
	                min_alpha=0.00025,
	                min_count=1,
	                dm=0)
	  
	model.build_vocab(tagged_data)

	for epoch in range(max_epochs):
	    print('iteration {0}'.format(epoch))
	    model.train(tagged_data,
	                total_examples=model.corpus_count,
	                epochs=model.iter)
	    # decrease the learning rate
	    model.alpha -= 0.0002
	    # fix the learning rate, no decay
	    model.min_alpha = model.alpha

	model.save("d2v.model")
	print("Model Saved")


plots, tags_movies = read_data('../data/movies_metadata.csv')
plot_list_tokenzied = tokenize_plots(plots)
doc_2_vec_model(plot_list_tokenzied, tags_movies)
