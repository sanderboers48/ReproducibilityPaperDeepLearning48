# -*- coding: utf-8 -*-
"""
@author: Abenezer
"""
# check
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ggplot')
import os
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from keras.utils.vis_utils import plot_model

import torch
from torch.utils.data import DataLoader
from torch import nn


'Read the file'

# 'Checking the data distribution per class'

'Checking the data distribution per class'
# df['Class'].value_counts().plot(kind='bar', title='Number of data point per class',color='C1')
# plt.ylabel('Data Points')
# plt.xlabel('Classes')

KIA = True
MODEL_TYPE = ["Original", "TOM", "PYTORCH"][2]
TRAINING = False
TEST_NOISE = False
SAVE_MODEL = False #overwrites earlier saved model!!
EPOCHS = 150
NUM_CLASSES = 10
NOISE_FACTOR = 2.5
MASK_FACTOR = 0.6

if KIA:
    NUM_FEATURES = 52
    df = pd.read_csv("../Datasets/Driving Data(KIA SOUL)_(150728-160714)_(10 Drivers_A-J).csv")
else:
    NUM_FEATURES = 21
    df = pd.read_csv("../Datasets/VehicularData(anonymized).csv")

def pre_process_encoder(df):
    print("Original dataframe size: ", df.shape)

    if(KIA):
        df_10 = df[df.Class.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])]
        print("Dataframe size (4 drivers): ", df_10.shape)
        mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9}
        df_10 = df_10.replace({'Class': mapping})

        # 'Features and label'
        X = df_10.drop('Class', 1)
        y = df_10.Class
        # print(y)
    else: #if not kia dataset
        # tom en sander meuk komt nu! :)
        df = df.iloc[85095:, [1, 10, 11, 12, 13, 14, 16, 19, 20, 22, 26, 27, 29, 31, 32, 33, 35, 36, 38, 40, 41, 42]]
        'Features and label'
        X = df.drop('Person_Id', 1)
        y = df.Person_Id
        print(y)


    print("X data shape (features): ", X.shape)
    print("y data shape (output classes): ", y.shape)

    if "Car_Id" in X.columns:
        X.drop('Car_Id', axis=1, inplace=True)
    if 'Trip' in X.columns:
        X.drop('Trip', axis=1, inplace=True)

    # TOM addition
    X = np.array(X)

    X = X[:, :NUM_FEATURES]  # reduce number of features to fit model input layer
    # X = X[:, :21]
    # dit is ook beetje beun, moeten ws ff goed kiezen welke 21 features we houden
    # nu zijn t gewoon de eerste 21

    return X, y


X, y = pre_process_encoder(df)

'Split the data set into window samples'

from sklearn.preprocessing import LabelEncoder


def window(X1, y1):
    X_samples = []
    y_samples = []

    encoder = LabelEncoder()
    encoder.fit(y1)
    y1 = encoder.transform(y1)

    length = 16
    overlapsize = length // 2
    n = y1.size

    Xt = np.array(X1)
    yt = np.array(y1).reshape(-1, 1)

    # for over the 263920  310495in jumps of 64
    for i in range(0, n, length - overlapsize):
        # grab from i to i+length
        sample_x = Xt[i:i + length, :]
        if (np.array(sample_x).shape[0]) == length:
            X_samples.append(sample_x)

        sample_y = yt[i:i + length]
        if (np.array(sample_y).shape[0]) == length:
            y_samples.append(sample_y)

    return np.array(X_samples), np.array(y_samples)


'for the label Select the maximum occuring value in the given array'


def max_occuring_label(sample):
    values, counts = np.unique(sample, return_counts=True)
    ind = np.argmax(counts)

    return values[ind]


'Creating y_sample label by taking only the maximum'


def label_y(y_value):
    y_samples_1 = []
    for i in range(len(y_value)):
        y_samples_1.append(max_occuring_label(y_value[i]))

    return np.array(y_samples_1).reshape(-1, 1)


from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils



def rnn_dimension(X, y):
    X_samples, y_samples = window(X, y)
    y_samples = label_y(y_samples)  #

    print("X samples window shape: ", X_samples.shape)  # (num_windows, samples per window, features per sample)?
    print("y samples shape: ", y_samples.shape)

    # Shuffling
    from sklearn.utils import shuffle
    X_samples, y_samples = shuffle(X_samples, y_samples)

    # to catagory
    # y_samples_cat = tf.keras.utils.to_categorical(y_samples)
    y_samples_cat = np_utils.to_categorical(y_samples)


    X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(X_samples, y_samples_cat, train_size=0.85)
    X_train, y_train = shuffle(X_train_rnn, y_train_rnn)

    return X_train, y_train, X_test_rnn, y_test_rnn  # train is shuffled, test not


X_train_5, y_train_5, X_test_5, y_test_5 = rnn_dimension(X, y)  # don't know why _5..?
# but this X data is 3 dimensional: (windows, sampels per window, features per sample)


device_lib.list_local_devices()


def normalizing(X_test):
    dim1 = X_test.shape[1]
    dim2 = X_test.shape[2]

    X_test_2d = X_test.reshape(-1, dim2)
    scale = StandardScaler()
    scale.fit(X_test_2d)

    X_test_scaled = scale.transform(X_test_2d)
    X_test_scaled = X_test_scaled.reshape(-1, dim1, dim2)

    return X_test_scaled

clean_model = load_model('Model_clean_binary_cross_ICTAI_vehicle2_1.h5')
# clean_model = load_model('Model_FCNN_ICTAI_vehicle2_1')
print(clean_model.summary())

print("X_train shape: ", X_train_5.shape)
print("Normalizing LSTM train/test data")
X_train_normalized = normalizing(X_train_5)

# X_test_normalized = normalizing(X_test_5) #dont do before adding noise??
X_test_normalized = np.copy(X_test_5) #only scale X_test after adding noise!

print("X_train shape: ", X_train_normalized.shape)
print("X_test shape: ", X_test_normalized.shape)
print("y_train shape: ", y_train_5.shape)
print("y_train shape: ", y_train_5.shape)
print("y_test shape: ", y_test_5.shape)

if MODEL_TYPE=='TOM':
    TOM_model = tf.keras.Sequential()
    TOM_model.add(tf.keras.layers.Input(shape=(None,NUM_FEATURES)))
    TOM_model.add(tf.keras.layers.LSTM(160, input_shape=(None,NUM_FEATURES), return_sequences=True))
    TOM_model.add(tf.keras.layers.BatchNormalization())
    TOM_model.add(tf.keras.layers.Dropout(.2))
    TOM_model.add(tf.keras.layers.LSTM(120, input_shape=(NUM_FEATURES,)))
    TOM_model.add(tf.keras.layers.BatchNormalization())
    TOM_model.add(tf.keras.layers.Dropout(.2))
    TOM_model.add(tf.keras.layers.Dense(NUM_CLASSES))
    TOM_model.add(tf.keras.layers.Softmax())
    print(TOM_model.summary())
    lstm_model = tf.keras.models.clone_model(TOM_model)
    lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                       loss=tf.keras.losses.BinaryCrossentropy(),
                       metrics=['accuracy'])

elif MODEL_TYPE == "PYTORCH":
    class LSTMmodel(torch.nn.Module):

        def __init__(self):
            super(LSTMmodel, self).__init__()

            self.lstm1 = nn.LSTM(hidden_size=160, input_size=NUM_FEATURES)
            self.batchNorm1 = nn.LazyBatchNorm1d()
            self.dropout1 = nn.Dropout(p=0.2)
            self.lstm2 = nn.LSTM(hidden_size=120, input_size=160)
            self.batchNorm2 = nn.BatchNorm1d(num_features=16)
            self.dropout2 = nn.Dropout(p=0.2)
            self.linear1 = nn.LazyLinear(NUM_CLASSES)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x, _ = self.lstm1(x)
            # x = self.batchNorm1(x)
            x = self.dropout1(x)
            x, _ = self.lstm2(x)
            x = self.batchNorm2(x)
            x = self.dropout2(x)
            x = self.linear1(x)
            x = self.softmax(x)
            return x

        def evaluate(self, X_test, y_test):
            with torch.no_grad():
                loss_fn = nn.BCELoss()
                self.eval()
                y_pred = self.forward(torch.from_numpy(X_test).double())
                loss = loss_fn(y_pred, torch.from_numpy(y_test).long())
                print(loss)

            return loss



    lstm_model = LSTMmodel()
    print(lstm_model)


else:
    lstm_model = tf.keras.models.clone_model(clean_model)
    lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                       loss=tf.keras.losses.BinaryCrossentropy(),
                       metrics=['accuracy'])



if MODEL_TYPE=='TOM':
    if TRAINING:
        print("----------------------")
        print("training lstm model")
        lstm_model.fit(X_train_normalized, y_train_5, epochs=EPOCHS)
        if SAVE_MODEL:
            lstm_model.save("group48_model")
    else:
        print("----------------------")
        print("loading group48 lstm model")
        lstm_model = tf.keras.models.load_model("group48_model")
else:
    if TRAINING:
        print("----------------------")
        print("training lstm model")
        lstm_model.fit(X_train_normalized, y_train_5, epochs=EPOCHS) #fit on the papers pretrained model
    else:
        print("----------------------")
        print("loading paper's lstm model")
        pass #use the paper's pretrained model




# # score = clean_model.evaluate(X_test_normalized, y_test_5, batch_size=50)
# score = lstm_model.evaluate(X_test_normalized, y_test_5, batch_size=50)
# # score = lstm_model.evaluate(X_test_5, y_test_5, batch_size=50)
# print('Test loss:', score[0])
# print('Test accuracy:', score[-1])



anomality_level = [0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]


# anomality_level = [0.05,0.1,0.2,0.4,0.6]

def LSTM_anomality(X_test_rnn, y_test_rnn):
    acc_noise_test = []
    acc_noise_test_rf_box = []
    for anomaly in anomality_level:
        print("=" * 5)
        print("for anomaly percentage = ", anomaly)

        def anomality(X, ): 
            orgi_data = np.copy(X_test_rnn.reshape(-1,NUM_FEATURES))
            print(orgi_data.shape)

            mask = np.random.choice(orgi_data.shape[0], int(len(orgi_data) *MASK_FACTOR), replace=False) # original mask. shape (samples*0.5,)
            # mask = np.random.choice(orgi_data.shape[0], int(len(orgi_data) * 1), replace=False)  # self-made mask shape (samples*0.5,)
            # print(orgi_data.shape)
            # print(mask.shape)
            # orgi_data[mask].shape




            if TEST_NOISE: # use noise instead of anomality
                std = anomaly * NOISE_FACTOR #to scale to a maximum noise std of 2
                noise_vector = np.random.normal(loc=0, scale=std, size=(orgi_data.shape[0], NUM_FEATURES))
                orgi_data = orgi_data + noise_vector
            else:
                orgi_data[mask] = orgi_data[mask] + orgi_data[mask] * anomaly
            return orgi_data

        def normalizing(X_test):

            dim1 = X_test.shape[1]
            dim2 = X_test.shape[2]

            X_test_2d = X_test.reshape(-1, dim2)
            scale = StandardScaler()
            scale.fit(X_test_2d)

            X_test_scaled = scale.transform(X_test_2d)
            X_test_scaled = X_test_scaled.reshape(-1, dim1, dim2)

            return X_test_scaled

        iter_score = []
        for i in range(5):
            X_test_rnn_anomal = np.copy(anomality(X_test_rnn).reshape(-1, X_test_rnn.shape[1], X_test_rnn.shape[2]))
            X_test_rnn_noise_scaled = normalizing(X_test_rnn_anomal)

            # pd.DataFrame(noising2(X_train.reshape(-1,49)))[1].head(1000).plot(kind='line')

            # score_1 = lstm_model.evaluate(X_test_rnn_anomal, y_test_rnn, batch_size=50, verbose=0) # not scaled
            score_1 = lstm_model.evaluate(X_test_rnn_noise_scaled, y_test_rnn, batch_size=50, verbose=0) # original
            print(score_1)
            if MODEL_TYPE != 'PYTORCH':
                score_1 = lstm_model.evaluate(X_test_rnn_anomal, y_test_rnn, batch_size=50, verbose=0)
            else:
                pass  # PyTorch evaluate here

            print(score_1)
            iter_score.append(score_1[1])  # accuracy
        #             print(score_1[1])

        #sorry i don't know what happened here - TOM
        dif = max(iter_score) - min(iter_score)
        score_2 = sum(iter_score) / len(iter_score)
        acc_noise_test.append(score_2)
        print('Avg Test loss:', score_1[0])  # this has to be wrong, right?
        print('Avg Test accuracy:', score_2)
        acc_noise_test_rf_box.append(dif)

    return acc_noise_test, acc_noise_test_rf_box


LSTM_acc_noise_test, LSTM_noise_acc_box = LSTM_anomality(X_test_normalized, y_test_5)
acc = []
fig1 = plt.figure()
for n in range(len(LSTM_acc_noise_test)):
    acc.append(LSTM_acc_noise_test[n])

plt.plot(anomality_level, acc)
plt.errorbar(anomality_level, LSTM_acc_noise_test, LSTM_noise_acc_box, fmt='.k', color='black', ecolor='red',
             elinewidth=3, capsize=0)


def normalizing_2d(X):
    scale = StandardScaler()
    scale.fit(X)

    X = scale.transform(X)

    return X


def anomality_2d(X, anomaly):
    X = np.array(X).reshape(-1, NUM_FEATURES)

    mask = np.random.choice(X.shape[0], int(len(X) * MASK_FACTOR), replace=False) #original masking

    # mask = np.random.choice(X.shape[0], int(len(X) * 1), replace=False) #mask up to all samples

    if TEST_NOISE:
        std = anomaly*NOISE_FACTOR
        noise_vector = np.random.normal(loc=0, scale=std, size=(X.shape[0], NUM_FEATURES))
        X = X + noise_vector
    else:
        X[mask] = X[mask] + X[mask] * anomaly

    return X


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, shuffle=False)

from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler

# TOM beun begint weer
y_dummy = np_utils.to_categorical(y)

from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle
from keras import layers
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_dummy, train_size=0.85)

X_train_scaled = np.copy(normalizing_2d(X_train))

X_train, X_test_, y_train, y_test_ = train_test_split(X_train_scaled, y_train, train_size=0.99)
X_train, y_train = shuffle(X_train, y_train)

mlp = Sequential()
mlp.add(Dense(160, input_dim=X_train.shape[1], activation='relu'))
mlp.add(layers.BatchNormalization())
mlp.add(layers.Dropout(0.5))
mlp.add(Dense(120, activation='relu'))
mlp.add(layers.BatchNormalization())
mlp.add(Dense(y_test.shape[1], activation='softmax'))
# mlp.add(Dense(1,activation='sigmoid'))
# mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("-----------------------")
print("train the FCNN")
mlp.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
with tf.device('/GPU:0'):
    mlp_history = mlp.fit(X_train, y_train, epochs=10, batch_size=100, shuffle=True)

X_test_normalized = normalizing_2d(X_test)

score = mlp.evaluate(X_test_normalized, y_test, batch_size=50)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


def mlp_acc_test(X_test, y_test):
    acc_noise_test = []
    acc_noise_test_rf_box = []

    #     anomality_level = [0,0.2,0.4,0.6,0.8,1]

    for anomal in anomality_level:

        i = 0
        iter_score = []
        while i < 5:
            X_test_anomal = np.copy(anomality_2d(X_test, anomal))
            X_test_normalized = normalizing_2d(X_test_anomal)

            score_1 = mlp.evaluate(X_test_normalized, y_test, batch_size=50)
            iter_score.append(score_1[1])
            i += 1
        #             print(i)

        dif = max(iter_score) - min(iter_score)
        score_2 = sum(iter_score) / len(iter_score)
        acc_noise_test.append(score_2)
        print('Avg Test loss:', score_2)
        print('Avg Test accuracy:', score_2)
        acc_noise_test_rf_box.append(dif)

    return acc_noise_test, acc_noise_test_rf_box


mlp_noise_acc, mlp_noise_acc_box = mlp_acc_test(X_test, y_test)
acc_mlp = []

for n in range(len(mlp_noise_acc)):
    acc_mlp.append(mlp_noise_acc[n])

# plt.plot(noise_sig,acc_mlp)
plt.plot(anomality_level, acc_mlp)
plt.errorbar(anomality_level, acc_mlp, mlp_noise_acc_box, fmt='.k', color='black',
             ecolor='red', elinewidth=3, capsize=0)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)

X_train_scaled = np.copy(normalizing_2d(X_train))

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


def acc_noise_test_dt(X_train, y_train, X_test, y_test):
    dt = DecisionTreeClassifier()
    print("-----------------------")
    print("train the Decision Tree")
    dt.fit(X_train, y_train)

    acc_noise_test_dt = []
    acc_noise_test_rf_box = []

    for anomal in anomality_level:

        iter_score = []
        for i in range(10):
            X_test_anomal = np.copy(anomality_2d(X_test, anomal))
            X_test_normalized = normalizing_2d(X_test_anomal)

            'Decision Tree'
            y_pred_dt = dt.predict(X_test_normalized)
            acc_n = metrics.accuracy_score(y_test, y_pred_dt)

            iter_score.append(acc_n)

        dif = max(iter_score) - min(iter_score)
        score_2 = sum(iter_score) / len(iter_score)
        acc_noise_test_dt.append(score_2)
        print('Avg Test loss:', score_2)
        print('Avg Test accuracy:', score_2)
        acc_noise_test_rf_box.append(dif)

    return acc_noise_test_dt, acc_noise_test_rf_box


dt_noise_acc, dt_noise_acc_box = acc_noise_test_dt(X_train_scaled, y_train, X_test, y_test)
acc_dt = []
# anomality_level = [0,0.2,0.4,0.6,0.8,1]
for n in range(len(dt_noise_acc)):
    acc_dt.append(dt_noise_acc[n])

plt.plot(anomality_level, acc_dt)
plt.errorbar(anomality_level, acc_dt, dt_noise_acc_box, fmt='.k', color='black',
             ecolor='red', elinewidth=3, capsize=0)

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def acc_noise_test_rf(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=20)
    print("-----------------------")
    print("train the Random Forest")
    rf.fit(X_train, y_train)

    acc_noise_test_rf = []
    acc_noise_test_rf_box = []

    for anomal in anomality_level:

        iter_score = []
        for i in range(10):
            X_test_anomal = np.copy(anomality_2d(X_test, anomal))
            X_test_normalized = normalizing_2d(X_test_anomal)

            'Random Forest'
            y_pred_rf = rf.predict(X_test_normalized)
            acc_n = metrics.accuracy_score(y_test, y_pred_rf)
            iter_score.append(acc_n)
        #             print(acc_n)

        dif = max(iter_score) - min(iter_score)
        acc_noise_test_rf_box.append(dif)
        score_2 = sum(iter_score) / len(iter_score)
        acc_noise_test_rf.append(score_2)

        print("=")
        print(score_2)

    return (acc_noise_test_rf, acc_noise_test_rf_box)


rf_noise_acc, rf_noise_acc_box = acc_noise_test_rf(X_train_scaled, y_train, X_test, y_test)
acc_rf = []

# anomality_level = [0,0.2,0.4,0.6,0.8,1]
for n in range(len(rf_noise_acc)):
    acc_rf.append(rf_noise_acc[n])

# plt.plot(noise_sig,acc_rf,'or')
# plt.plot()
plt.plot(anomality_level, acc_rf)
plt.errorbar(anomality_level, acc_rf, rf_noise_acc_box, fmt='.k', color='black',
             ecolor='red', elinewidth=3, capsize=0)
# plt.boxplot(noise_sig,rf_noise_acc_box)


# mpl.style.use('seaborn-poster')
fig2 = plt.figure()
# anomality_level = [0,0.2,0.4,0.6,0.8,1]
# noise_sig = anomality_level 
noise_level = np.array(anomality_level[:10])*NOISE_FACTOR

if TEST_NOISE:
    plt.axis([-0.07, noise_level[-1]+0.1, 0, 1.08])
    plt.plot(noise_level, acc[:10], marker='^', label="LSTM", linewidth=3.5)
    plt.plot(noise_level, acc_mlp[:10], marker='o', label="FCNN", linewidth=3.5)
    plt.plot(noise_level, acc_dt[:10], marker='*', label="Decision Tree", linewidth=3.5)
    plt.plot(noise_level, acc_rf[:10], marker='x', label="Random Forest", linewidth=3.5)
    plt.xlabel("STD of sensor noise induced in the data ", fontsize=16)
else:
    plt.axis([-0.07, .82, 0, 1.08])
    plt.plot(anomality_level[:10], acc[:10], marker='^', label="LSTM", linewidth=3.5)
    plt.plot(anomality_level[:10], acc_mlp[:10], marker='o', label="FCNN", linewidth=3.5)
    plt.plot(anomality_level[:10], acc_dt[:10], marker='*', label="Decision Tree", linewidth=3.5)
    plt.plot(anomality_level[:10], acc_rf[:10], marker='x', label="Random Forest", linewidth=3.5)
    plt.xlabel("Percentage of sensor anomalities induced in the data (*100)", fontsize=12)
plt.ylabel("accuracy", fontsize=20)
# plt.title("Accuracy on noisy data")
plt.legend(loc=3, fontsize=16)
# plt.grid()
plt.show()
