#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/hbaflast/bert-sentiment-analysis-tensorflow/blob/master/bert_sentiment_analysis_tensorflow.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Bert Sentiment Analysis - TensorFlow

# ## Import

# In[ ]:


# Install missing librairies
get_ipython().system('pip install transformers')
# Switch to tf2 (Colab run tensorflow 1.X by default for the moment)
get_ipython().run_line_magic('tensorflow_version', '2.x')


# In[ ]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, Input

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import (BertTokenizer, BertForSequenceClassification, TFBertForSequenceClassification,
                          CamembertTokenizer, CamembertForSequenceClassification, TFCamembertForSequenceClassification)


# In[ ]:


plt.rcParams["figure.figsize"] = (16, 9)


# In[4]:


# check gpu
get_ipython().system('nvidia-smi')


# ## Load data

# ### Kaggle credentials

# In[ ]:


# enter your Kaggle credentionals here
os.environ['KAGGLE_USERNAME'] = ""
os.environ['KAGGLE_KEY'] = ""


# In[6]:


get_ipython().system('kaggle datasets download hbaflast/french-twitter-sentiment-analysis')


# ### Read dataset

# In[ ]:


df_dataset = pd.read_csv("/content/french-twitter-sentiment-analysis.zip", sep=',')


# In[ ]:


df_dataset = df_dataset.sample(frac=0.1, random_state=42)  # sample to speed-up computation


# In[9]:


df_dataset.info()


# In[10]:


df_dataset['label'].value_counts().plot.bar();


# ## Text length distribution

# In[11]:


df_dataset['sent_len'] = df_dataset['text'].apply(lambda x: len(x.split(" ")))
max_seq_len = np.round(df_dataset['sent_len'].mean() + 2 * df_dataset['sent_len'].std()).astype(int)
max_seq_len


# In[12]:


df_dataset['sent_len'].plot.hist()
plt.axvline(x=max_seq_len, color='k', linestyle='--', label='max len');


# ## Load tokenizer

# In[13]:


tokenizer = CamembertTokenizer.from_pretrained('camembert-base')


# In[14]:


tokenizer.tokenize("J'aime bien faire des achats en ligne")


# In[15]:


tokenizer.encode("J'aime bien faire des achats en ligne")


# ## Load transformers model

# In[16]:


transformers_model = TFCamembertForSequenceClassification.from_pretrained('jplu/tf-camembert-base', num_labels=2)


# In[17]:


transformers_model.summary()


# ## Process input example

# In[18]:


input_ =  tf.expand_dims(tokenizer.encode("J'aime bien faire des achats en ligne"), 0)
input_


# In[19]:


att_mask = tf.expand_dims(np.ones(input_.shape[1], dtype='int32'), 0)
att_mask


# In[20]:


logits = transformers_model([input_, att_mask])
logits


# ## Pre-processing

# ### Tokenize text & padding

# In[21]:


df_dataset.head()


# In[22]:


input_sequences = []
# The attention mask is an optional argument used when batching sequences together.
# The attention mask is a binary tensor indicating the position of the padded indices so that the model does not attend to them.
attention_masks = []

for text in tqdm_notebook(df_dataset['text']):
    sequence_dict = tokenizer.encode_plus(text, max_length=max_seq_len, pad_to_max_length=True)
    input_ids = sequence_dict['input_ids']
    att_mask = sequence_dict['attention_mask']

    input_sequences.append(input_ids)
    attention_masks.append(att_mask)


# In[23]:


print(input_sequences[0])
print(attention_masks[0])


# In[ ]:


labels = df_dataset['label'].values


# In[25]:


print(labels[0])


# ## Train Test Split

# In[ ]:


X_train, X_test, y_train, y_test, att_masks_train, att_masks_test = (
    train_test_split(input_sequences, labels, attention_masks, random_state=42, test_size=0.2)
)


# In[ ]:


X_train = tf.constant(X_train)
X_test = tf.constant(X_test)

y_train = tf.constant(y_train)
y_test = tf.constant(y_test)

att_masks_train = tf.constant(att_masks_train)
att_masks_test = tf.constant(att_masks_test)


# In[28]:


print(f'Train | X shape: {X_train.shape}, att_mask shape: {att_masks_train.shape}, y shape: {y_train.shape}')
print(f'Test | X shape: {X_test.shape}, att_mask shape: {att_masks_test.shape}, y shape: {y_test.shape},')


# ## Create model

# In[ ]:


def create_model():
    model = TFCamembertForSequenceClassification.from_pretrained('jplu/tf-camembert-base', num_labels=2)
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam(lr=2e-5)
  
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'])
  
    return model


# In[30]:


model = create_model()
model.summary()


# ## Training

# In[31]:


loss, metric = model.evaluate([X_test, att_masks_test], y_test, batch_size=32, verbose=0)
print(f"Loss before training: {loss:.4f}, Accuracy before training: {metric:.2%}")


# In[32]:


history = model.fit([X_train, att_masks_train], y_train, batch_size=32, epochs=2, validation_data=([X_test, att_masks_test], y_test))


# ## Test model on new sentences

# In[ ]:


def predict(text):
    # pre-process text
    encoded_text = tokenizer.encode(text)

    input_ = tf.expand_dims(encoded_text, 0)

    logits = model(input_)[0][0]
    pred = tf.nn.softmax(logits).numpy()
    
    return pred


# In[ ]:


text = "Qu'est c'que c'est trop beau la vie d'artiste"


# In[35]:


predict(text)


# In[ ]:


text = "Je n'aime pas faire la vaisselle"


# In[37]:


predict(text)


# ## Save model
# 

# In[ ]:


save_path = "finetuned-model"


# In[ ]:


os.mkdir(save_path)


# In[ ]:


model.save_pretrained(save_path)


# In[ ]:





# In[ ]:




