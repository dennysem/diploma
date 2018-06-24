import numpy as np
import warnings
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution1D
from keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings('ignore')

if __name__ == "__main__":

    print ("Reading data...")

    data = pd.read_json("./datasets/data1.json")
    X = data['query']
    y = data['tags']
    data = pd.DataFrame({'query': X, 'tags': y})
    #data = shuffle(data, random_state=5)

    data = data.head(15000)

    X = data['query']
    y = data['tags']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=5)

    print("Prepairing data...")

    def padd_data(data, max_len=29, value=0, mode='post'):
        return pad_sequences(data, maxlen=max_len, value=value, padding=mode, dtype='str')

    all_tags = ['B-filter', 'B-predefined_filter', 'stop_word', 'B-dimension',
                'B-filter_value', 'I-predefined_filter', 'B-metric', 'I-metric', 'I-dimension', 'I-filter']

    X_train = padd_data(X_train, max_len=29, value=0)
    X_test = padd_data(X_test, max_len=29, value=0)
    y_train = padd_data(y_train, max_len=29, value=0)
    y_test = padd_data(y_test, max_len=29, value=0)
    y_train[y_train == '0.0'] = 'stop_word'
    y_test[y_test == '0.0'] = 'stop_word'

    #print (X_train.shape)
    a = list(set(X_train.reshape((X_train.shape[0] * X_train.shape[1]))))
    b = list(set(X_test.reshape((X_test.shape[0] * X_test.shape[1]))))
    a = list(set(a + b))

    word2numb = dict(zip(sorted(a), list(range(len(a)))))
    numb2word = dict(zip(list(range(len(a))), sorted(a)))
    tag2numb = dict(zip(sorted(all_tags), list(range(len(all_tags)))))
    numb2tag = dict(zip(list(range(len(all_tags))), sorted(all_tags)))

    n_classes = len(all_tags)
    n_vocab = len(word2numb)
    print (n_classes, n_vocab)

    print("Building model...")

    model = Sequential()
    model.add(Embedding(n_vocab, 100))
    model.add(Convolution1D(128, 4, padding='same', activation='relu'))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
    model.compile('rmsprop', 'categorical_crossentropy')

    model.load_weights("best_weights")


    def predict_one_query(query, model):
        query = padd_data([query.split()], value=0)[0]
        query = np.array(list(map(lambda x: word2numb[x], query)))
        pred = model.predict_on_batch(query[np.newaxis, :])
        pred = np.argmax(pred, -1)[0]
        return list(map(lambda x: numb2tag[x], pred))

    while (True):
        query = input('Enter query: ')
        query = query.strip()
        try:
            print(predict_one_query(query, model)[0:len(query.split())])
        except:
            KeyError

