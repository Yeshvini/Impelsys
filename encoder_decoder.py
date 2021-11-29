# Packages for encoder-decoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np

max_story_len=2000
max_summary_len=200

seq_df = pd.read_csv('clean_data.csv')
#Select the Summaries and Text between max len defined above
seq_df = seq_df.loc[(seq_df['story_word_count'] < 1500) & (seq_df['summary_word_count'] < 200)]
seq_df['summary'] = '_START_ ' + seq_df['summary'].astype(str) + '_STOP_ '
seq_df['summary'].iloc[0]
seq_df.reset_index(drop=True,inplace=True)
#Split the data to TRAIN and VALIDATION sets
x_tr,x_val,y_tr,y_val=train_test_split(np.array(seq_df['story']),np.array(seq_df['summary']),test_size=0.1,random_state=0,shuffle=True)

#Lets tokenize the text to get the vocab count, a tokenizer for stories on training data
x_tokenizer = Tokenizer() 
x_tokenizer.fit_on_texts(list(x_tr))

def rare_words_analysis(x_tokenizer):
  cnt,tot_cnt,freq,tot_freq= (0,)*4
  thresh = 4
  for key,value in x_tokenizer.word_counts.items():
      tot_cnt=tot_cnt+1
      tot_freq=tot_freq+value
      if(value<thresh):
          cnt=cnt+1
          freq=freq+value
  print(" % of rare words in vocabulary:",(cnt/tot_cnt)*100)
  print("Total Coverage of rare words:",(freq/tot_freq)*100)
  return tot_cnt,cnt

print('Rare words analysis of stories')
tot_cnt,cnt = rare_words_analysis(x_tokenizer)

def lstm_data_prep(xy_tr,xy_val):

  #prepare a tokenizer for stories on training data
  xy_tokenizer = Tokenizer(num_words=tot_cnt-cnt) 
  xy_tokenizer.fit_on_texts(list(xy_tr))

  #convert text sequences into integer sequences (i.e one-hot encodeing all the words)
  xy_tr_seq    =   xy_tokenizer.texts_to_sequences(xy_tr) 
  xy_val_seq   =   xy_tokenizer.texts_to_sequences(xy_val)

  #padding zero upto maximum length
  xy_tr    =   pad_sequences(xy_tr_seq,  maxlen=max_story_len, padding='post')
  xy_val  =   pad_sequences(xy_val_seq, maxlen=max_story_len, padding='post')

  #size of vocabulary ( +1 for padding token)
  xy_voc   =  xy_tokenizer.num_words + 1

  print("Size of Vocabulary - {}".format(xy_voc))

  return xy_tr,xy_val,xy_voc

x_tr,x_val,x_voc = lstm_data_prep(x_tr,x_val)

# Rare words analysis of summaries
#Lets tokenize the text to get the vocab count, a tokenizer for stories on training data
y_tokenizer = Tokenizer() 
y_tokenizer.fit_on_texts(list(y_tr))
print('Rare words analysis of summaries')
tot_cnt,cnt = rare_words_analysis(y_tokenizer)
y_tr,y_val,y_voc  = lstm_data_prep(y_tr,y_val)

'''
Return Sequences = True: When the return sequences parameter is set to True, LSTM produces the hidden state and cell state for every timestep
Return State = True: When return state = True, LSTM produces the hidden state and cell state of the last timestep only
Initial State: This is used to initialize the internal states of the LSTM for the first timestep
Stacked LSTM: Stacked LSTM has multiple layers of LSTM stacked on top of each other. This leads to a better representation of the sequence
'''

from keras import backend as K 
K.clear_session()

def stacked_lstm_encoder(max_story_len):
    latent_dim = 300
    embedding_dim=200

    # Encoder
    encoder_inputs = Input(shape=(max_story_len))

    #embedding layer
    enc_emb =  Embedding(x_voc, embedding_dim,trainable=True)(encoder_inputs)

    #encoder lstm 1
    encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

    #encoder lstm 2
    encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

    #encoder lstm 3
    encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))

    #embedding layer
    dec_emb_layer = Embedding(y_voc, embedding_dim,trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
    decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

    #dense layer
    decoder_dense =  TimeDistributed(Dense(y_voc, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model 
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    print(short_model_summary)

    return model , short_model_summary

model = stacked_lstm_encoder(max_story_len)

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)
#history=model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:] ,epochs=1,callbacks=[es],batch_size=128, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))
