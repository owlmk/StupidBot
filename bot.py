
# coding: utf-8

# In[2]:
from flask import Flask, request, redirect
import re
import pandas as pd
pd.set_option('display.max_colwidth', 200)
    


# In[3]:


df = pd.read_csv('train.csv')
#df


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

vectorizer = TfidfVectorizer(ngram_range = (1, 3))
vec = vectorizer.fit_transform(df['questions'].values.astype('U'))

@app.route('/', methods = ['GET', 'POST'])
# In[12]:


#df.iloc[rsi]['answers']


# In[18]:


def get_response():
	input_str = request.values.get('Body')
	def get_response(q):
		my_q = vectorizer.transform([q])
		cs = cosine_similarity(my_q, vec)
		rs = pd.Series(cs[0]).sort_values(ascending = bool(0))
		rsi = rs.index[0]
		return df.iloc[rsi]['answers']
	if input_str:
		return str(get_response(input_str))
	else:
		return 'something bad'
app.run()