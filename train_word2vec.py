import time, unidecode, json
import regex as re
import pandas as pd
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
import gensim
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE

from bokeh.io import output_notebook
from bokeh.plotting import show, figure


class TrainWord2Vec(object):

	def __init__(self):
		self.stopwords = stopwords.words('english')
		self.stopwords = [word for word in self.stopwords if word not in ['not','no', 'nor']]
		print("\n number of stopwords --- ", len(self.stopwords))
		self.normalize_mapping = json.load(open("data/normalize_mapping.json"))

	def clean_text(self, text): # unused
		try:
			text = self.normalizeMapping(text)
			# text = re.sub(r"[^A-Za-z0-9@#$]", " ", text)
			# text = re.sub(r"[^A-Za-z0-9]", " ", text)
			text = re.sub(r"[^A-Za-z]", " ", text)
			# text = re.sub(r"\s{3,}", " ", text)
			
			text = self.removeStopWords(text)
			# text = self.getLemma(text)
			text = re.sub(r"\s+", " ", text)
			text = " ".join([word for word in word_tokenize(text) if len(word) > 2])
			text = self.handle_unicode(text)
		
		except Exception as e:
			print("\n Error in clean_str --- ",e)
			print("\n text with error --- ",text, type(text))

		return text.lower().strip()

	def handle_unicode(self, val):
		return unidecode.unidecode(val)

	def removeStopWords(self, text):
		return " ".join([token for token in text.split() if token not in self.stopwords])
			
	def checkLemma(self, wrd):
		return nltk.stem.WordNetLemmatizer().lemmatize(nltk.stem.WordNetLemmatizer().lemmatize(wrd, 'v'), 'n')

	def getLemma(self, text):
		return " ".join([self.checkLemma(tok) for tok in text.lower().split()])


	def load_data_and_labels(self):
		"""
		Loads MR polarity data from files, splits the data into words and generates labels.
		Returns split sentences and labels.
		"""
		# Load data from files
		positive_examples = list(open('data/rt-polarity.pos', "r").readlines())
		positive_examples = [s.strip() for s in positive_examples]
		negative_examples = list(open('data/rt-polarity.neg', "r").readlines())
		negative_examples = [s.strip() for s in negative_examples]
		# Split by words
		x_text = positive_examples + negative_examples
		submission = defaultdict(list)
		submission['text'].extend(x_text)
		data = pd.DataFrame(submission)
		data['text'] = data['text'].apply(self.clean_text)
		return data['text']

	def readData(self):
		try:
			data  = pd.read_csv('data/text_classification_dataset.csv')
			print('\n train size == ', len(data))
			data['reviews'] = data['reviews'].apply(self.clean_text)
			print("\n data['reviews'] --- ", len(data['reviews']), len(data['labels']))
		except Exception as e:
			print("\n Error in readData --- ", e,"\n",traceback.format_exc())
		return data['reviews']

	def normalizeMapping(self, query):
		try:
			splited_query = query.split()
			for index, word in enumerate(splited_query):
				word = word.lower()
				if word in self.normalize_mapping:
					splited_query[index] = self.normalize_mapping[word]
			query = " ".join(splited_query)
			return query
		except Exception as e:
			print("\n Error in normalizeMapping --- ",traceback.format_exc())
			return query

	def reduceDimension(self, model):
		# reduce vector dimentionality using T-SNE (t distributed stochastic neighbour embedding)
		# - tehnique for high dimensionality reduction, particularly suited for the visualization of high dimensional data in 2D

		tsne = TSNE(n_components=2, n_iter=250)
		x = model[model.wv.vocab]
		print("\n orig x shape --- ", x.shape)
		x_2d = tsne.fit_transform(x)
		print("\n transformed data shape --- ", x_2d.shape)

		coord_df = pd.DataFrame(x_2d, columns=['x', 'y'])
		coord_df['token'] = model.wv.vocab.keys()
		# print("\n model.wv.vocab.keys() --- ", model.wv.vocab.keys())
		# print("\n coord_df.head --- ", coord_df.head)

		# plot the graph
		coord_df.plot.scatter('x', 'y', figsize=(8,8), marker=0, s=10, alpha=0.2)


	def trainWord2Vec(self, sentences):

		# size = embeddings dimesion
		# sg = 1-skipgram, 0-cbow
		# window = context words
		# min_count = ignore the words with frequency < 5
		# workers = number of worker threads
		# seed = the answer to the universe life nd everything

		start_time=time.time()
		model = Word2Vec(sentences = sentences, size=32, window=3, sg=1, min_count=1, workers=4, seed=42)
		print("\n word2vec training time --- ", time.time() - start_time)
		print("\n not --- ", model['not'])
		print("\n most similar to not --- ",model.most_similar('not'))
		print("\n most similar to good --- ",model.most_similar('good'))
		# print(model.similarity('mother', 'father'))
		# print(model.most_similar(positive=['father', 'woman'], negative=['man']))
		model.save('imdb_dataset_word2vec.model')
		return model

	def main(self):
		# data = self.readData()
		sentences = self.load_data_and_labels()
		sentences = [word_tokenize(sent) for sent in sentences]
		print("\n number of sentences --- ", len(sentences))
		print("\n sentences --- ",sentences[:3])

		model = self.trainWord2Vec(sentences)
		# self.reduceDimension(model)


if __name__ == '__main__':
	obj = TrainWord2Vec()
	obj.main()