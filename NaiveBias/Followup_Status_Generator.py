"""
This program will automatically generate the follow-up status 
from enquiry followup comment.
Programs written by Ratheesh.T.S (ratheeshthana@gmail.com)
You are free to use or modify this program.

"""
import numpy as np
import pandas as pd
import string
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#Data preprocessing
follow_ups=pd.read_csv('enquiry_followups.csv')
follow_ups.describe()
follow_ups=follow_ups.dropna()
follow_ups=follow_ups.apply(lambda x: x.astype(str).str.lower())
follow_ups['Status'].loc[(follow_ups['Status'] == 'nuetral')|(follow_ups['Status'] == 'nutral')]='neutral'
follow_ups['Status']=follow_ups['Status'].apply(lambda x: getIndexOfStatus(x))
status_count=follow_ups.groupby('Status').count()
remove_punk_dic= dict((ord(p_value),None) for p_value in string.punctuation)
lemmer=nltk.stem.WordNetLemmatizer()
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = lem_normalize)
text_counts= cv.fit_transform(follow_ups['Comment'].tolist())#Output status 
#this is how we split the data as train set and test set
X_train, X_test, y_train, y_test = train_test_split(text_counts, follow_ups['Status'].tolist(), test_size=0.2, random_state=4)
#Import scikit-learn metrics module for accuracy calculation
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
comment_list=['she wont come','he didnt pickup the call']
status_list=getPredict(comment_list)


""" Functions """
#input status values return index
def getIndexOfStatus(status):
    status_values=['cancelled','negative','neutral','positive','registered']
    return status_values.index(status)
	
#lemmatize tokens
def lem_tokens(tokens):
    return [lemmer.lemmatize(token_word) for token_word in tokens]

""" This function will lemmatize input_corpus
1. converts input_corpus to lower letter
2. remove punctuation
3. lemmatize using lem_tokens
""" 
def lem_normalize(input_corpus):
   return lem_tokens(nltk.word_tokenize(input_corpus.lower().translate(remove_punk_dic)))

"""
This method input comment as list 
para data_list: comments as list ['comment','comment2']
return result status value
"""
def getPredict(data_list):
 cv2 = CountVectorizer(vocabulary=cv.vocabulary_,lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = lem_normalize)
 X_test1= cv2.transform(data_list)
 predicted2= clf.predict(X_test1)
 status_values=['cancelled','negative','neutral','positive','registered']
 Status_List = [status_values[i] for i in predicted2]
 return Status_List
