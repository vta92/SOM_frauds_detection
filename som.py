import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom

#fraud = outliers of applicants, far from neighborhoods

dataset=pd.read_csv("Credit_Card_Applications.csv")

x = dataset.iloc[:,:-1].values #all but the last column as indep variables
y = dataset.iloc[:,-1].values #the last column


#note:unsupervised learning so we won't use y for training
scaler = MinMaxScaler(feature_range=(0,1))
x = scaler.fit_transform(x) #should we even transform the ID?


#10x10 grid output with 15 dim input and sigma of 1
som = MiniSom(x=10,y=10, input_len=15,sigma=1.0, learning_rate=0.2)
som.random_weights_init(x)
som.train_random(x, num_iteration=200)


from pylab import bone, pcolor, colorbar, plot, show
bone()

#using colors instead of bones to show the map
pcolor(som.distance_map().T) #distance_map will return a matrix of mean interneuron distances (transposed)
colorbar() #legend, intensity of color. Dark = low mid, light = high mid, frauds

#to tell if the customers associated to the light color = approved or not
markers = ['o', 's'] #circles and squares. O for non-approval, and s for approval
colors = ['r', 'b']

for i,j in enumerate(x):
    #get the winning node in the customer i
    winning_node = som.winner(j)
    plot(winning_node[0]+0.5, winning_node[1]+0.5, markers[y[i]],
         markeredgecolor=colors[y[i]], markerfacecolor='None', markersize=10, markeredgewidth=2)
    #plot(winning_node[1]+0.5)
    #plot(markers[y[i]]) #if the customer was approved or not, based on y vector of our dataset
    #plot(markeredgecolor = colors[y[i]], markerfacecolor = None, markersize = 10, markeredgewidth = 3)
show()

map = som.win_map(x) #get the dictionary of all the winning nodes in our som maps

#manually from the graph, winning nodes have coordinates: (2,2), (5,1)
#open the map object and input the whitest coordinates

frauds = np.concatenate((map[(2,2)],map[5,1]), axis=0)
frauds = np.vstack(frauds)    
frauds = scaler.inverse_transform(frauds)


#todo: abstractize the functions and make it easier to see