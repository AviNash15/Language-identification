import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding, SpatialDropout1D
from keras.layers import GlobalMaxPool1D, GlobalAveragePooling1D
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
import time

print('-'*80)
print(time.ctime())
print("Reading Input Files")
train_df = pd.read_csv('/home/avinash/train.txt', sep="  ", names=['lang_labels', 'text'])
val_df   = pd.read_csv('/home/avinash/valid.txt', sep="  ", names=['lang_labels', 'text'])

train_df['lang_labels'] = train_df['lang_labels'].astype('str')
val_df['lang_labels'] = val_df['lang_labels'].astype('str')

x_train, y_train = train_df.text.values, train_df.lang_labels.values
x_test, y_test = val_df.text.values, val_df.lang_labels.values

num_classes = len(train_df.lang_labels.unique())
encoder = LabelEncoder() 
encoder = encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test) 

y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)
print('-'*80)
print(time.ctime())
print("Tokenizing")
tokens = Tokenizer()
tokens.fit_on_texts(x_train)

x_train = tokens.texts_to_sequences(x_train)
x_test = tokens.texts_to_sequences(x_test)

vocab_size = len(tokens.word_index)
print(len(tokens.word_index))

max_len = 15

print('-'*80)
print(time.ctime())
print("Padding Sequences")
EMBEDDING_DIM = 100
x_train  = pad_sequences(x_train, maxlen=max_len, padding='post', truncating='post', value=0.0)
x_test   = pad_sequences(x_test,  maxlen=max_len, padding='post', truncating='post', value=0.0)

x_train = np.fliplr(x_train)
x_test = np.fliplr(x_test)

print('Shape of data tensor:', x_train.shape)
print('Shape of label tensor:', y_train.shape)

model = Sequential()
model.add(Embedding(vocab_size,
                    EMBEDDING_DIM,
                    input_length=max_len,
                    trainable=True,
                    name="embedding"))
model.add(SpatialDropout1D(0.25))
model.add(GlobalAveragePooling1D())
model.add(BatchNormalization())
model.add(Dense(512, activation="tanh"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax', name="language_prediction"))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
             )

print('-'*80)
print(time.ctime())
print("Model Summary")
print(model.summary())

print('-'*80)
print(time.ctime())
print("Started Training")

checkpoint = ModelCheckpoint('temp.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=0, verbose=1, mode='auto')

history = model.fit(
    x_train,
    y_train,
    batch_size=512,
    epochs=10,
    validation_data=(x_test,y_test),
    verbose=1,
    callbacks = [checkpoint,early_stop],
    initial_epoch=0,
    shuffle=True)
