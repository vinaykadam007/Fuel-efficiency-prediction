import pandas as pd
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random

data = pd.read_csv("C:/Users/Anil/Desktop/ML data/PERIODIC_DATA/30Sec/163 device data 03-july-2018 to 07-july-2018-30sec.csv")


filter_1 = (data['OBU_ID']==1269) |(data['OBU_ID']==1270)|(data['OBU_ID']==1274)|(data['OBU_ID']==1275)|(data['OBU_ID']==1276)|(data['OBU_ID']==1277)|(data['OBU_ID']==1316)|(data['OBU_ID']==1317)|(data['OBU_ID']==1318)|(data['OBU_ID']==1319)|(data['OBU_ID']==1320) \
|(data['OBU_ID']==1321)|(data['OBU_ID']==1322)|(data['OBU_ID']==1323)|(data['OBU_ID']==1337)|(data['OBU_ID']==1339)|(data['OBU_ID']==1341)|(data['OBU_ID']==1348)|(data['OBU_ID']==1350)|(data['OBU_ID']==1359) \
|(data['OBU_ID']==1368)|(data['OBU_ID']==1369)|(data['OBU_ID']==1370)|(data['OBU_ID']==1375)|(data['OBU_ID']==1385)|(data['OBU_ID']==1389)|(data['OBU_ID']==1393)

data = data[filter_1]
data.drop(['OBU_ID','EVENT_UTC','ALTITUDE','HEADING','AIR_PRESSURE_ACTUAL1','AIR_PRESSURE_ACTUAL2','FUEL_LEVEL','VEHICAL_BATTERY_POTENTIAL','ENG_OIL_DIGITAL','ENG_OIL_ACTUAL','ENG_COOLANT_TEMP','FUEL_CONSUM','AIR_PRESSURE_DIGITAL','NO_OF_SAT','PACKET_STATUS','LOCATION', \
	'B.PACKET_TYPE','LOCAL_TIME_STAMP','LATITUDE','LONGITUDE','WHEEL_SPEED','RESERVE1','RESERVE2','KAFKA_TIME_STAMP','RECEIVED_TIMESTAMP','VERSION','IGNITION_STATUS','VEH_STATUS','INTERNAL_BATTERY_POTENTIAL','PACKET_CODE','CAL_ODO','GPS_ODO','VEHICLE_ODO'],axis=1,inplace=True)
data = data.dropna(subset=['GPS_SPEED','ENGINE_SPEED'])

print(data.head())

x= data['GPS_SPEED']
y = data['ENGINE_SPEED']



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

x_train = np.array(x_train).reshape(-1,1)
x_test = np.array(x_test).reshape(-1,1)
#print(y_train)
#print(x_train,x_test)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_features = x_train.shape[1]
n_neurons_1 = 512
n_neurons_2 = 256
n_neurons_3 = 128


X = tf.placeholder(dtype=tf.float32, shape=[None,n_features])
Y = tf.placeholder(dtype=tf.float32,shape=[None])

sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode='fan_avg',distribution = 'uniform',scale=sigma)
bias_initializer = tf.zeros_initializer()


w_hidden_1 = tf.Variable(weight_initializer([n_features, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))


w_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

w_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))


w_out = tf.Variable(weight_initializer([n_neurons_3, 1]))
bias_out = tf.Variable(bias_initializer([1]))


hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, w_hidden_1),bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, w_hidden_2),bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, w_hidden_3),bias_hidden_3))

out = tf.transpose(tf.add(tf.matmul(hidden_3,w_out),bias_out))

mse = tf.reduce_mean(tf.squared_difference(out,Y))
#mse = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
opt = tf.train.AdamOptimizer().minimize(mse)

batch_size = 100
epochs = 10
#init_lo = tf.local_variables_initializer()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())#initialize_all_variables()



	for e in range(epochs):
		shuffle_data = np.random.permutation(np.arange(len(y_train)))
		x_train = x_train[shuffle_data]
		y_train = y_train[shuffle_data]
		for i in range(0, len(y_train)//batch_size):
			start = i*batch_size
			batch_x = x_train[start:start + batch_size]
			batch_y = y_train[start:start + batch_size]
			_, c = sess.run([opt,mse], feed_dict = {X:batch_x, Y:batch_y})
			
		print('Epoch', e,'completed out of', epochs)
	
	#correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

    #accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
#	pred = sess.run(out, feed_dict={X: x_test})
#	y_pred = pred[0]
#	y_pred = pred[0] > 0.5
#	print(pred)

























