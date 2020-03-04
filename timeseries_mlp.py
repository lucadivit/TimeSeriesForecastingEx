import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input, Activation, Dropout
from sklearn.model_selection import train_test_split
import math, random
from sklearn.preprocessing import MinMaxScaler

def plot_chart(values, colors, title, x_label, y_label, legend, histories):
    vals = []
    for i in range(len(values)):
        vals.append([])
    for h in histories:
        for i in range(len(values)):
            vals[i] = vals[i] + h.history[values[i]]
    for i in range(len(values)):
        plt.plot(vals[i], colors[i])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(legend, loc="upper right")
    plt.show()
    return None


#I take only the second column
X = pd.read_csv('daily-min-temperatures.csv', usecols=[1], engine='python')
X = X.values
X = X.astype('float32')
#Data normalization. NB l'inverse transform funziona anche per 
#valori non presenti nel dataset (che rispettano il range)
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

#Crea una serie statica
look_back = 100
X_train, X_test = train_test_split(X, test_size=0.1, shuffle=False)
X_train, y_train = create_dataset(X_train, look_back=look_back)
X_test, y_test = create_dataset(X_test, look_back=look_back)

inputs = Input(shape=(look_back,))
inner = Dense(16)(inputs)
inner = Activation("relu")(inner)
inner = Dense(8)(inputs)
inner = Activation("relu")(inner)
inner = Dense(4)(inputs)
inner = Activation("relu")(inner)
outputs = Dense(1)(inner)
outputs = Activation("linear")(outputs)
model = Model(inputs=inputs, outputs=outputs)
model.compile("adam", loss="mean_squared_error")#, metrics=['mae']
history = model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=2, validation_data=(X_test, y_test))

train_score = model.evaluate(X_train, y_train)
test_score = model.evaluate(X_test, y_test)

#faccio anche la radice per il calcolo dell rooted mean squarred error
print('Train Score: %.2f MSE (%.2f RMSE)' % (train_score, math.sqrt(train_score)))
print('Test Score: %.2f MSE (%.2f RMSE)' % (test_score, math.sqrt(test_score)))
# generate predictions
testPredict = model.predict(X_test)
trainPredict = model.predict(X_train)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(X)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(X)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(X)-1, :] = testPredict


#Rescale Data and plotting
X = scaler.inverse_transform(X)
trainPredictPlot = scaler.inverse_transform(trainPredictPlot)
testPredictPlot = scaler.inverse_transform(testPredictPlot)
plt.plot(X)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
'''
#plot loss and val loss
histories = [history]
plot_chart(values=["loss", "val_loss"], colors=["g-", "r-"], title="Time Series", x_label="epochs", y_label="loss", legend=["train", "test"], histories=histories)
'''

'''
#Custom prediction
prova = []
for i in range(0, 100):
    prova.append(i)

#Create an array with one sample with 100 elems.
prova = np.array(prova).reshape(1, -1)
prova = scaler.transform(prova)
pr = model.predict(prova)
pr = scaler.inverse_transform(pr)
print(pr)
'''