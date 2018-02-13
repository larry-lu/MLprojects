
# coding: utf-8

# # Naive Bayes classification of news articles
# 
# Xiaoyu Lu
# 
"""In this assignment, we are going to use the Naive Bayes algorithm as a means to automatically classify 
news reports. In particular, we will build on the material that we presented in class and test the 
classifier’s performance using different settings
"""

from os import chdir, getcwd
from glob import glob
import pyspark
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
import time
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer

path = getcwd()
chdir(path)


# Below are two functions we will use during this assignment

def if_topic_in(topic, topic_list = topic_list):
    """function to determine if each entry belongs to our topic list
    ---------------------------------------------
    
    :param topic: list of many topics of one article
    :param topic_list: list of pre-defined topics
    
    :returns: index of first element in the topic list that belongs to topic_list
    """
    try:
        ans = list(set(topic).intersection(topic_list))
    except:
        ans = ""
    
    return ans


def cleanbody(text):
    """function to clean text by removing punctuations, and numbers
    ---------------------------------------------
    
    :param text: a string
    
    :returns: string with punctuations and numbers removed
    """
    stemmer = SnowballStemmer('english')
    text = text.replace('\n',' ').lower().strip()
    text = re.sub("[^a-z, A-Z]+", "", text).split()
    text = ' '.join(stemmer.stem(i) for i in text)
    stemmed = ' '.join([word for word in text.split() if word not in stopwords_set])
    return(stemmed)


# 1. Pre-processing the Reuters' dataset (Reuters-21578)

f_list = glob('reuters21578/*.sgm')

topic_list = ["money", "fx", "crude", "grain", "trade", "interest", "wheat", 
              "ship", "corn", "oil", "dlr", "gas", "oilseed", "supply", "sugar", 
              "gnp", "coffee", "veg", "gold", "soybean", "bop", "livestock", "cpi"]

doi_list = list()
for filename in f_list:
    print('Start parsing {0}...'.format(filename))
    file = open(filename, 'rb')
    soup = BeautifulSoup(file, 'html.parser')
    file.close()
    for topic_raw in soup.find_all('topics'):
        topic = topic_raw.get_text().split('-')
        topic = if_topic_in(topic)
        if len(topic) != 0:
            body = topic_raw.find_next('body').get_text()
            for t in topic:
                tb_tup = (t, body)
                doi_list.append(tb_tup)
    print('Finished parsing {0}'.format(filename))

data = pd.DataFrame(doi_list)
data.columns = (['topic', 'body'])
data['body'] = data['body'].apply(cleanbody)
print('A total number of {0} items were retrieved. Articles with multiple classes are recorded multiple times.'.format(len(data)))
data.head()

#saving the data as a txt file
data.to_csv('training_test_data.txt')
data.loc[0:10].to_csv('top10.txt')

# 2. TF-IDF transformation in sklearn

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import RegexpTokenizer

data = pd.read_csv('training_test_data.txt')

body_list = list(data['body'])
start = time.clock()
vectorizer = TfidfVectorizer()
vectorizer.fit(data['body'])
print ("sklearn TFIDF processing time: {0:.5f} s".format(time.clock() - start))

#sklearn TFIDF processing time: 0.50861 s

# 3. TF-IDF transformation in `pyspark`

from pyspark.sql import SparkSession
from pyspark.sql import Column
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.ml.feature import HashingTF, IDF, OneHotEncoder, StringIndexer
from pyspark.ml.classification import NaiveBayes

path = getcwd()
chdir(path)

spark = SparkSession        .builder        .appName("NewsClassification")        .getOrCreate()

df = spark.read.csv("training_test_data.txt",header=True,inferSchema=True)

stop_stem_udf = udf(stop_stem, ArrayType(StringType()))
df = df.withColumn("body", stop_stem_udf("body"))

#following section transforms the text using TFIDF
start = time.clock()
hashingTF = HashingTF(inputCol="body", outputCol="term_freq")
df = hashingTF.transform(df)
idf = IDF(inputCol="term_freq", outputCol="tfidf", minDocFreq=5)
idfModel = idf.fit(df)
df = idfModel.transform(df)
print ("pyspark TFIDF processing time: {0:.5f} s".format(time.clock() - start))
#pyspark TFIDF processing time: 0.01415 s

# 4. Building a Naive Bayes Classifier

#Using the OneHotEncoder to convert the topics into discrete integers
stringIndexer = StringIndexer(inputCol="topic", outputCol="topicIndex")
model = stringIndexer.fit(df)
indexed = model.transform(df)

val_dict = dict()
train_test_cv_split_params = {'50/40/10': [0.5, 0.4, 0.1],
                               '60/30/10': [0.6, 0.3, 0.1], 
                               '70/20/10': [0.7, 0.2, 0.1]}

for split_param in train_test_cv_split_params.keys(): #run the model for each train/test/cv split
    for seed in np.arange(10): #run each model 10 times using different random seed
        train,test,cv = indexed.select("tfidf","topicIndex").randomSplit(train_test_cv_split_params[split_param],seed=seed)

        #Naive bayes
        nb = NaiveBayes(featuresCol="tfidf", labelCol="topicIndex", predictionCol="NB_pred",
                        probabilityCol="NB_prob", rawPredictionCol="NB_rawPred")
        nbModel = nb.fit(train)
        cv = nbModel.transform(cv)
        total = cv.count()
        correct = cv.where(test['topicIndex'] == cv['NB_pred']).count()
        accuracy = correct/total
        val_dict[(split_param, seed)] = accuracy


params = max(val_dict, key = val_dict.get)
print("The combination of parameters that produced the highest accuracy ({0:.2f}): train/test/cv split ratio: {1}, randomseed: {2}".format(max(val_dict.values()),params[0], params[1]))


def meancal(val_dict, split_param):
    l = list()
    for i in val_dict.keys():
        if i[0] == split_param:
            l.append(val_dict[i])
    #print(l)
    return np.mean(l)

print('The mean accuracy of the 30 models: {0:.3f}'.format(np.mean(list(val_dict.values()))))

for split_param in train_test_cv_split_params:
    mean_accuracy = meancal(val_dict, split_param)
    print('The split condition {0} has a mean accuracy of {1:.3f}'.format(split_param, mean_accuracy))