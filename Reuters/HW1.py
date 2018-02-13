from os import chdir, getcwd
from glob import glob
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.snowball import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from pyspark.sql import SparkSession
from pyspark.sql import Column
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.ml.feature import HashingTF, IDF, OneHotEncoder, StringIndexer
from pyspark.ml.classification import NaiveBayes

path = getcwd()
chdir(path)

f_list = glob('reuters21578/*.sgm')

topic_list = ["money", "fx", "crude", "grain", "trade", "interest", "wheat", "ship", "corn", "oil", "dlr", 
"gas", "oilseed", "supply", "sugar", "gnp", "coffee", "veg", "gold", "soybean", "bop", "livestock", "cpi"]

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
    stemmer = PorterStemmer()
    text = text.replace('\n',' ').lower().strip()
    text = re.sub("[^a-z, A-Z]+", "", text)
    processed = ''.join(stemmer.stem(i) for i in text)
    return(processed)

#The following section saves the output as a list of (topic, body) tuples.
doi_list = list()
for filename in f_list:
    print('Start parsing {0}'.format(filename))
    file = open(filename, 'rb')
    soup = BeautifulSoup(file, 'html.parser')
    file.close()
    for topic_raw in soup.find_all('topics'):
        topic = topic_raw.get_text().split('-')
        topic = if_topic_in(topic)
        if len(topic) != 0:         #only get articles whos topic is within our list
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

#Save the DataFrame as a txt file
data.to_csv('training_test_data.txt')
#Save the top 10 items of the dataset
data.loc[0:10].to_csv('top10.txt')

###Naive Bayes Classifier on Pyspark
spark = SparkSession\
        .builder\
        .appName("NewsClassification")\
        .getOrCreate()

df = spark.read.csv("training_test_data.txt",header=True,inferSchema=True)
stopwords_set = set(stopwords.words('english'))
stemmer = PorterStemmer("english")

#This function removes stopwords (e.g. 'the', 'a') from the text
def stop_stem(tokens):
    tokens = tokens.split()
    stemmed = [word for word in tokens if word not in stopwords_set]
    return stemmed

stop_stem_udf = udf(stop_stem, ArrayType(StringType()))
df = df.withColumn("tokenized", stop_stem_udf("body"))

#following section transforms the text using TFIDF
start = time.clock()
hashingTF = HashingTF(inputCol="tokenized", outputCol="term_freq")
df = hashingTF.transform(df)
idf = IDF(inputCol="term_freq", outputCol="tfidf", minDocFreq=5)
idfModel = idf.fit(df)
df = idfModel.transform(df)
print ("pyspark TFIDF processing time: {0:.5f} s".format(time.clock() - start))

#Using the OneHotEncoder to convert the topics into discrete integers
stringIndexer = StringIndexer(inputCol="topic", outputCol="topicIndex")
model = stringIndexer.fit(df)
indexed = model.transform(df)

val_dict = dict()
train_test_val_split_params = {'50/40/10': [0.5, 0.4, 0.1],
                               '60/30/10': [0.6, 0.3, 0.1], 
                               '70/20/10': [0.7, 0.2, 0.1]}

for split_param in train_test_val_split_params.keys():
    for seed in np.arange(10):
        train,test,val = indexed.select("tfidf","topicIndex").randomSplit(train_test_val_split_params[split_param],seed=seed)

        #Naive bayes
        nb = NaiveBayes(featuresCol="tfidf", labelCol="topicIndex", predictionCol="NB_pred",
                        probabilityCol="NB_prob", rawPredictionCol="NB_rawPred")
        nbModel = nb.fit(train)
        val = nbModel.transform(val)
        total = val.count()
        correct = val.where(test['topicIndex'] == val['NB_pred']).count()
        accuracy = correct/total
        val_dict[(split_param, seed)] = accuracy
    
#get the max parameters that produced the highest accuracy
params = max(val_dict, key = val_dict.get)
print("The combination of parameters that produced the highest accuracy: train/test/cv split ratio: {0}, randomseed: {1}".format(params[0], params[1]) )