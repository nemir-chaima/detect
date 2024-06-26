print('helllloo ')
import pandas as pd
print('pandas')
from sklearn.model_selection import train_test_split
print('sktlearn')

import streamlit as st
print("streamlit")


import tensorflow as tf
print('tensorflow as tf')
from transformers import TFBertModel, BertTokenizer
print('from transformers import TFBertModel, BertTokenizer')
from tensorflow.keras.layers import Input, Dense, Flatten
print('from tensorflow.keras.layers import Input, Dense, Flatten')
from tensorflow.keras.models import Model
print('from tensorflow.keras.models import Model')
print("le script est lancé ")


df = pd.read_csv('/home/grp/FilterAI/detect/cleaned_df.csv', sep=',')
colonnes_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# creer une colonne labels ( elle nest finalement pas necessaire!!!! a revoir )
df['label'] = df.apply(lambda row: [row[col] for col in colonnes_labels], axis=1)
Texts = df['comment_text']

# charger le model et le tockenizer 
bert_model = TFBertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# tockenisation et prep de données 
max_sequence_length = 128
def preprocess_text_data(text_data, max_sequence_length):
    input_ids = []
    for text in text_data:
        tokens = tokenizer(text, padding='max_length', truncation=True, max_length=max_sequence_length, return_tensors="tf")['input_ids'][0]
        input_ids.append(tokens)
    return tf.convert_to_tensor(input_ids)

text_tokenized = preprocess_text_data(Texts, max_sequence_length)

# faire le data split
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
labels_df = df.loc[:, labels]
X_train, X_test, y_train, y_test = train_test_split(text_tokenized.numpy(), labels_df.values, test_size=0.2, random_state=42)

num_classes = 6
# utilser bert pour les embeding 
input_text = Input(shape=(max_sequence_length,), dtype=tf.int32)
bert_output = bert_model(input_text)[0]
# pour freeze les couches restantes 
for layer in bert_model.layers:
    layer.trainable = False

# rajouter une classe de classification qui prends les embeding de bert commme entree
# qui retourne la classe predite (vecteur des probabilité)
x = Dense(256, activation='tanh')(bert_output)
x = Flatten()(Dense(128, activation='tanh')(x))

output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=input_text, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# entrainer le model
epochs = 20
history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.1, shuffle=True, batch_size=200)

# sauvegarder le model pour le tester apres 
save_directory = '/home/grp/FilterAI/detect'  # le chemin ou on peut save le model pre entrainé
os.makedirs(save_directory, exist_ok=True)
model.save(save_directory)




print( "le model est saved ")