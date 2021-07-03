

import pandas as pd
import numpy as np
import csv
import re
from BanglaUtilites import bangla_text_preprocess
import pickle
from keras.models import load_model




authentic0 = pd.read_csv('Data/Authentic-48K.csv')
#authentic_news = bangla_text_preprocess(authentic0['content'])
#print("Number of authentic news: ", len(authentic_news)) 

fake0 = pd.read_csv('Data/Fake-1K.csv')
#fake_news = bangla_text_preprocess(fake0['content'])
#print("Number of fake news: ", len(fake_news)) 
#print(fake.shape)



#with open("Data/authenticNews.pickle", "wb") as fp: 
#    pickle.dump(authentic_news, fp)

#with open("Data/fakeNews.pickle", "wb") as fp: 
#    pickle.dump(fake_news, fp)





with open("Data/authenticNews.pickle", "rb") as fp:   # Unpickling
    authentic_news = pickle.load(fp)
print("Number of authentic news: ", len(authentic_news))


with open("Data/fakeNews.pickle", "rb") as fp:   # Unpickling
    fake_news = pickle.load(fp)
print("Number of fake news: ", len(fake_news))


# # Using all news, create a tokenizer object

# Total Fake news = 1299, 
# Total Authentic news = 48678



#allNews = []
#allNews.extend(authentic_news) 
#allNews.extend(fake_news)
#print("Total number of news for tokenization: " ,len(allNews))


#from keras.preprocessing.text import Tokenizer
#max_words = 15000 # considers only the top max_words number of words in the dataset 
#tokenizer = Tokenizer(num_words = max_words)
#tokenizer.fit_on_texts(allNews) 



# saving
#with open('Data/tokenizer.pickle', 'wb') as handle:
#    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



# loading the tokenizer
with open('Data/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# # Build the Secondary Dataframe



for i in range(len(authentic0)):
    authentic0['content'] = authentic_news[i]


for i in range(len(fake0)):
    fake0['content'] = fake_news[i]


# # Function for running the Model


from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt

#maxlen = 100 # cut sentences after maxlen words
#training_samples = 1000 # train on this number of samples
#validation_samples = 500 # validates on this number of samples
#max_words = 15000 # considers only the top max_words number of words in the dataset 
#embedding_dim = 256  # length of embedding vector for each word

#tokenizer = Tokenizer(num_words = max_words)
#tokenizer.fit_on_texts(sentences)

def run_model(data, j, maxlen = 150, embedding_dim = 256, epochs = 10, batch=32, max_words = 15000): 
    l = len(data)
    #training_samples = int(l*0.6) 
    #validation_samples = int(l*0.2) 
    sentences = data['content']
    sequences = tokenizer.texts_to_sequences(sentences)
    word_index = tokenizer.word_index 
    #print('Found unique tokens = ', len(word_index))
    labels = np.asarray(data['label'])
    data = pad_sequences(sequences, maxlen = maxlen)
    #print('Shape of data tensor = ', data.shape)
    #print('Shape of label tensor = ', labels.shape)
    
    auth_data = data[:l-1299]
    fake_data = data[l-1299:]
    auth_label = labels[:l-1299]
    fake_label = labels[l-1299:]
    
    indices = np.arange(fake_data.shape[0])
    fake_data = fake_data[indices]
    fake_label = fake_label[indices]
    
    cut1 =  int(len(auth_data)*0.7) 
    cut2 =  int(len(fake_data)*0.7)
    #print("Cut1, Cut2 = " , cut1, cut2)
    #auth_tr = auth_data[:int(len(auth_data*.7))]
    #fake_tr = fake_data[:int(len(fake_data*.7))]
    
    #print(auth_tr.shape)
    
    data_tr = np.concatenate((auth_data[:cut1], fake_data[:cut2]))
    data_va =  np.concatenate((auth_data[cut1:], fake_data[cut2:]))
    labels_tr = np.concatenate((auth_label[:cut1], fake_label[:cut2]))
    labels_va =  np.concatenate((auth_label[cut1:], fake_label[cut2:]))
    
    indices = np.arange(data_tr.shape[0])
    np.random.shuffle(indices)
    x_train = data_tr[indices]
    y_train = labels_tr[indices]
    
    indices = np.arange(data_va.shape[0])
    np.random.shuffle(indices)
    x_val = data_va[indices]
    y_val = labels_va[indices]

    #x_train = data[:training_samples]
    #y_train = labels[:training_samples]
    #x_val = data[training_samples: training_samples+validation_samples]
    #y_val = labels[training_samples: training_samples+validation_samples]

    print(' Shape of train data tensor = ', x_train.shape)
    print(' Shape of validation data tensor = ', x_val.shape)
    #print(' labels of training data = ', y_train)
    print(' Training Set : ')
    print(' \t Authentic Data  = ', sum(y_train))
    print(' \t Fake Data      = ', len(y_train) - sum(y_train))
    #print(' labels of validation data = ', y_val)
    print(' Validation Set : ')
    print(' \tAuthentic Data = ', sum(y_val))
    print(' \tFake Data      = ', len(y_val) - sum(y_val))
    
    from keras.models import Sequential, Model
    from keras.layers import Embedding, Flatten, Dense, Input, LSTM


    input_tensor = Input(shape=(x_train.shape[1],))
    x = Embedding(max_words, embedding_dim, input_length=maxlen)(input_tensor)
    #x = LSTM(128, return_sequences = True,  dropout=0.2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output_tensor = Dense(1, activation='sigmoid')(x)
    model = Model(inputs = input_tensor, outputs = output_tensor)

    model.summary()
    #from tensorflow.keras.utils import plot_model
    #mdl = "result/model"+str(j)+".png"
    #plot_model(model, show_shapes=True, to_file=mdl)
    
    model_file = "result/model"+str(j)+".h5"
    
    callback_list = [
        ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience = 3),
        ModelCheckpoint ( filepath= model_file, monitor='val_accuracy', save_best_only = True),
        TensorBoard(log_dir = 'my_log_dir', histogram_freq = 1, embeddings_freq =1)
    ]

    model.compile(optimizer = 'rmsprop',
                 loss = 'binary_crossentropy',
                 metrics = ['accuracy'])
    
    history = model.fit(x_train, y_train,
                       epochs = epochs,
                       batch_size = batch,
                       callbacks = callback_list,
                       validation_data = (x_val, y_val))

    #x_test = data[training_samples+validation_samples: ]
    #y_test = labels[training_samples+validation_samples: ]

    #print(model.evaluate(x_test, y_test))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc)+1)

    plt.plot(epochs, acc, 'b-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'g-', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b-', label='Training loss')
    plt.plot(epochs, val_loss, 'g-', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()

    plt.show()
    j = j + 1






# # Segment the dataset and run on model



l = len(authentic0)
print(l)

import random
idx = [0]
for i in range(1,5):
    a = random.randint(3000, 15000)
    #print(a)
    idx.append(idx[i-1]+a)
idx.append(len(authentic0))
print(idx)


dataset_cuts = [0, 9670, 21591, 35152, 39121, 48678]



for i in range(4):
    #print(idx[i])
    data = pd.concat([authentic0[idx[i]:idx[i+1]], fake0])
    print(" Loop ",i+1," \n Dataset Size: ", len(data))
    run_model(data, i)



data = pd.concat([authentic0[idx[4]:idx[5]], fake0])
sentences = data['content']
sequences = tokenizer.texts_to_sequences(sentences)
word_index = tokenizer.word_index 
labels = np.asarray(data['label'])
data = pad_sequences(sequences, maxlen = 150)
    
    
auth_data = data[:l-1299]
fake_data = data[l-1299:]
auth_label = labels[:l-1299]
fake_label = labels[l-1299:]
    
indices = np.arange(fake_data.shape[0])
x_test = np.concatenate((auth_data, fake_data))
y_test = np.concatenate((auth_label, fake_label))

print(' Test Set : ')
print(' \t Authentic Data  = ', sum(y_test))
print(' \t Fake Data      = ', len(y_test) - sum(y_test))

model = load_model('result/model2.h5')
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=32)
print("test loss, test acc:", results)


# In the unknown dataset, it gives 100% accuracy. 





