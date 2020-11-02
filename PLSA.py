#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize


# ### Understanding the Tfidftransformer usage

# In[9]:


doc = [
        "A video of government workers in Puducherry hurriedly throwing the body of a COVID-19 positive man into a pit has caused massive outrage, prompting the administration to order a probe into the incident.",
        "The COVID pandemic has hit the world and the Vedanta Group business. It has incurred losses in oil, gas and mining sectors.",
        "Yuvraj Singh was diagnosed with a cancerous tumor in his left lung following India's World Cup triumph in 2011. He had scored 362 runs and claimed 15 wickets in the tournament and was bestowed with the Player of the Tournament award in the end"
]


# In[10]:


cv = CountVectorizer()
word_count_vector = cv.fit_transform(doc)


# In[11]:


word_count_vector.shape


# In[15]:


feature_names = cv.get_feature_names()


# In[14]:


tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
tf_idf_vector = tfidf_transformer.fit_transform(word_count_vector)


# In[17]:


query = tf_idf_vector[2]
df = pd.DataFrame(query.T.todense(), index=feature_names,columns=['tfidf'])
df.sort_values(by=['tfidf'],ascending=False)


# In[19]:


word_count_vector.toarray()


# ## Topic Modelling on Practical Dataset

# In[21]:


df = pd.read_csv('abcnews-date-text.csv',error_bad_lines=False)


# In[22]:


df.head(20)


# In[23]:


data_text = df[['headline_text']].astype('str')
data_text.shape


# In[24]:


data_text = data_text.loc[1:100000,:]


# In[25]:


data_text.shape


# In[26]:


stopw = stopwords.words('english')
stopw


# In[27]:


def stopwords_remove(x):
    terms = x.split()
    terms = [w for w in terms if w not in stopw]
    sentence = ' '.join(terms)
    return sentence


# In[28]:


data_text['Refined_headlines'] = data_text['headline_text'].apply(lambda x: stopwords_remove(x))


# In[29]:


data_text.head()


# In[30]:


def word_count(x):
    terms = x.split()
    return len(terms)
data_text['word_count']=data_text['Refined_headlines'].apply(lambda x: word_count(x))


# In[31]:


data_text.head()


# In[32]:


data_text['word_count'].describe()


# In[36]:


fig = plt.figure(figsize=(10,5))

plt.hist(
        data_text['word_count'],
        bins=10,
        color='#60505C'
)

plt.title("Distribution- Article word count",fontsize=16)
plt.ylabel("frequency",fontsize=12)
plt.xlabel("Word count",fontsize=12)

plt.show()


# In[39]:


import seaborn as sns
sns.set_style('darkgrid')

fig = plt.figure(figsize=(4,9))

sns.boxplot(
        data_text['word_count'],
        orient='v',
        width=0.5,
        color='#ff8080'
)

plt.ylabel("Word Count",fontsize=12)
plt.title("Distribution- Article word count",fontsize=16)

plt.show()


# In[40]:


headline_sentences = [''.join(text) for text in data_text['Refined_headlines']]


# In[41]:


vectorizer = CountVectorizer(analyzer='word',max_features=5000)
x_counts = vectorizer.fit_transform(headline_sentences)


# In[46]:


x_counts.toarray().shape


# In[48]:


transformer = TfidfTransformer(smooth_idf=False)
x_tfidf = transformer.fit_transform(x_counts)


# In[49]:


x_tfidf


# In[62]:


num_topics = 5
model = NMF(n_components = num_topics, init='nndsvd')
model.fit(x_tfidf)


# In[63]:


def get_nmf_topics(model, n_top_words):
    
    feat_names = vectorizer.get_feature_names()
    
    word_dict = {}
    for i in range(num_topics):
        
        words_ids = model.components_[i].argsort()[:-n_top_words-1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic #'+'{:02d}'.format(i+1)] = words
        
    return pd.DataFrame(word_dict)


# In[64]:


get_nmf_topics(model,10)


# In[60]:


model.components_[0][0]


# In[ ]:




