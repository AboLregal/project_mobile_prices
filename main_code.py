# -*- coding: utf-8 -*-
"""
Created on Thu May  9 08:38:32 2024

@author: Mr XIII
"""

import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df=pd.read_csv('train - train.csv')
df.fillna(0,inplace=True)
X = df.drop(columns=['price_range'])
X['Area']=np.round(X['px_height']*X['px_width'])
col=list(X.columns)

for i in col:
    plt.figure()
    sns.boxplot(x=df['price_range'],y=X[i])
    plt.show()
    
X2=X[['talk_time','battery_power','fc','m_dep',
      'mobile_wt','n_cores','ram','Area']].values
Y=df['price_range'].values
x1_train, x1_test, y1_train, y1_test = train_test_split(X2, Y, test_size=0.3, random_state=42)
#%%
# Create the neural network
net = MLPC(hidden_layer_sizes=(30,20,15), 
          activation='identity',
          max_iter=10000, tol=1e-10, verbose=100,
          learning_rate_init=0.01, early_stopping=True,
          solver='adam',
          learning_rate='invscaling',
          n_iter_no_change=2000)

# Train the network
net.fit(x1_train, y1_train)

# Make predictions
y12 = np.round(net.predict(X2)).astype(int)
y12[y12 < 0] = 0
y12[y12 > 3] = 3
# y22 = np.round(net.predict(x2)).astype(int)

# Calculate the accuracy
ef1 = np.sum(y12 == Y) / len(Y)
print('Total Efficiency = ',ef1*100)

# Calculate confusion matrix
cm = confusion_matrix(Y, y12)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1','2','3'], yticklabels=['0','1','2','3'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for all data with eff='+str(ef1*100))
plt.show()

# Saving Model
filename = 'finalized_model__9265.sav'
pickle.dump(net, open(filename, 'wb'))