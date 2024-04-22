print('helllloo ')
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
import streamlit as st

print("le script est lancé ")


df = pd.read_csv('/home/grp/FilterAI/detect/cleaned_df.csv', sep=',')
colonnes_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# creer une colonne labels ( elle nest finalement pas necessaire!!!! a revoir )
df['label'] = df.apply(lambda row: [row[col] for col in colonnes_labels], axis=1)
Texts = df['comment_text']

# charger le model et le tockenizer 



print( "le model est saved ")