import pandas as pd

data = pd.read_csv("005930.KS.csv")

xdata = []
ydata = []



for i, rows in data.iterrows() :
    xdata.append([rows['Date'],rows['Open'],rows['Close']])

print(xdata)

for i in range(len(xdata)) :
    if int(xdata[i][1]) - int(xdata[i][2]) < 0 :
        ydata.append(1)
    elif int(xdata[i][1]) - int(xdata[i][2]) == 0 :
        ydata.append(0.5)
    elif int(xdata[i][1]) - int(xdata[i][2]) > 0 :
        ydata.append(0)

data['UD'] = ydata
print(data)
data.to_csv('test.csv',index=False)