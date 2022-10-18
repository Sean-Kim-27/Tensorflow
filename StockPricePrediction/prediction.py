import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()

data = pd.read_csv("test.csv")
data1 = pd.read_csv("test.csv")

#x = data.values.astype(float)
#x_scaled = sc.fit_transform(x)
#data = pd.DataFrame(x_scaled,columns=data.columns)

xdata = []
ydata = data1['UD'].values
zdata = data['Close'].values
date = []

xdata1 = []
zdata1 = data1['Close'].values

for i, rows in data1.iterrows() :
    xdata1.append([rows['Open'],rows['Close']])
    date.append([rows['Date']])
    
for i, rows in data.iterrows() :
    xdata.append([rows['Open'],rows['Close']])
    

#sc1 = MinMaxScaler()
#xdata = sc.fit_transform(xdata)
#zdata = sc1.fit_transform(zdata.values.reshape(-1,1))
print(xdata[0:4])
print(zdata[1:5])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16,activation='tanh'),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(64,activation='tanh'),
    tf.keras.layers.Dense(64,activation='tanh'),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(16,activation='tanh'),
    tf.keras.layers.Dense(1,activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience = 10)

#history = model.fit(np.array(xdata[:-50]),np.array(ydata[:-50]),epochs=1000, callbacks=[early_stop])
history = model.fit(np.array(xdata[:-1]),np.array(ydata[1:]),epochs=1000, callbacks=[early_stop])

#
#import matplotlib.pyplot as plt
#plt.figure(figsize=(12, 4))
#
#plt.subplot(1,2,1)
#plt.plot(history.history['loss'], 'b-', label='loss')
#
#plt.xlabel('Epoch')
#plt.legend()
#
#plt.subplot(1,2,2)
#plt.plot(history.history['accuracy'], 'g-', label='accuracy')
#plt.xlabel('Epoch')
#plt.ylim(0, 1)
#plt.legend()
#
#plt.show()
#
#
#prd = model.predict([[xdata[-50:]]])
#print(xdata[-50:])
#print(prd)
prd = model.predict([xdata[-1]])
if prd[0][0] < 0.5 :
    prd = '하락'
elif prd[0][0] == 0.5 :
    prd = '일정'
elif prd[0][0] > 0.5 :
    prd = '상승'

for i in reversed(range(1,6)) :
    if ydata[-i] == 1 :
        print(date[-i]," = 시가 : ",xdata1[-i][0]," 종가 : ",xdata1[-i][1]," => 주가 상승")
    elif ydata[-i] == 0.5 :
        print(date[-i]," = 시가 : ",xdata1[-i][0]," 종가 : ",xdata1[-i][1]," => 주가 일정")
    elif ydata[-i] == 0 :
        print(date[-i]," = 시가 : ",xdata1[-i][0]," 종가 : ",xdata1[-i][1]," => 주가 하락")
print("다음날 예상 등락은 ",prd," 입니다.")
