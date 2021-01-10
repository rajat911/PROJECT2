#!/usr/bin/env python
# coding: utf-8

# In[17]:


# Importing necessary libraries and packages
import numpy as np
import pandas as pd
import itertools
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[18]:


# Loading data as numpy array from csv database files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test = test.set_index('id', drop = True)


# In[19]:


# Counting number of rows and columns in the data
print('Shape of Training Data: ', train.shape)

# Gettiing a hang of the data in each column and their names
print('\n \n TRAIN \n', train.head())
print('\n \n TEST \n', test.head())

# Looking for any places where training data has NaN values
print('\n \nNumber of Null values in Train Set: ', train['text'].isna().sum())
print('Number of Null values in Test Set: ', test['text'].isna().sum())

# Dropping all rows where text column is NaN
train.dropna(axis=0, how="any", thresh=None, subset=['text'], inplace=True)
test = test.fillna(' ')


# In[20]:


# Checking length of each article
length = []
[length.append(len(str(text))) for text in train['text']]
train['length'] = length

print(min(train['length']), max(train['length']), round(sum(train['length'])/len(train['length'])))


# In[21]:


# Minimum length is 1. We need to spot some outliers and get rid of them. Counting how many outliers are there
print(len(train[train['length'] < 50]))

# Skimming through such short texts just to be sure
print(train['text'][train['length'] < 50])

# Removing outliers, it will reduce overfitting
train = train.drop(train['text'][train['length'] < 50].index, axis = 0)

print(min(train['length']), max(train['length']), round(sum(train['length'])/len(train['length'])))


# In[22]:


# Secluding labels in a new pandas dataframe for supervised learning
train_labels = train['label']

# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(train['text'], train_labels, test_size=0.1, random_state=0)


# In[23]:


# Setting up Term Frequency - Inverse Document Frequency Vectorizer
tfidf = TfidfVectorizer(stop_words = 'english', max_df = 0.7)

# Fit and transform training set and transform test set
tfidf_train = tfidf.fit_transform(x_train) 
tfidf_test = tfidf.transform(x_test)
tfidf_test_final = tfidf.transform(test['text'])


# In[24]:


# Setting up Passive Aggressive Classifier
pac = PassiveAggressiveClassifier(max_iter = 50)

# Fitting on the training set
pac.fit(tfidf_train, y_train)

# Predicting on the test set
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score * 100, 2)}%')


# In[25]:


# Creating confusion matrix with columns as True Positive, False Negative, False Positive and True Negative 
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, range(2), range(2))
sn.set(font_scale=1)
sn.heatmap(df_cm, annot=True, annot_kws={'size':14}, fmt='d').set_title('Confusion Matrix')
plt.show()

# Creating classification report
print('\nClassification Report: \n', classification_report(y_test, (y_pred > 0.5)))


# In[12]:


# Only for submission on kaggle. Please ignore.
#test_pred = pac.predict(tfidf_test_final)

#submission = pd.DataFrame({'id':test.index, 'label':test_pred})
#print(submission.shape, submission.head())

#submission.to_csv('submission.csv', index = False)


# In[ ]:




