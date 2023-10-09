import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb 
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
# We have imported all dependencies
 # Drop Adj close and volume feature

data = pd.read_csv("C:/Users/Anil/Desktop/ML data/PERIODIC_DATA/06Sec/163 device data 01-july-2018-06sec.csv")


filter_1 = (data['OBU_ID']==1269) |(data['OBU_ID']==1270)|(data['OBU_ID']==1274)|(data['OBU_ID']==1275)|(data['OBU_ID']==1276)|(data['OBU_ID']==1277)|(data['OBU_ID']==1316)|(data['OBU_ID']==1317)|(data['OBU_ID']==1318)|(data['OBU_ID']==1319)|(data['OBU_ID']==1320) \
|(data['OBU_ID']==1321)|(data['OBU_ID']==1322)|(data['OBU_ID']==1323)|(data['OBU_ID']==1337)|(data['OBU_ID']==1339)|(data['OBU_ID']==1341)|(data['OBU_ID']==1348)|(data['OBU_ID']==1350)|(data['OBU_ID']==1359) \
|(data['OBU_ID']==1368)|(data['OBU_ID']==1369)|(data['OBU_ID']==1370)|(data['OBU_ID']==1375)|(data['OBU_ID']==1385)|(data['OBU_ID']==1389)|(data['OBU_ID']==1393)

data = data[filter_1]
# data.drop(['FUEL_CONSUM','GPS_ODO','OBU_ID','EVENT_UTC','ALTITUDE','HEADING','AIR_PRESSURE_ACTUAL1','AIR_PRESSURE_ACTUAL2','FUEL_LEVEL','VEHICAL_BATTERY_POTENTIAL','ENG_OIL_DIGITAL','ENG_OIL_ACTUAL','ENG_COOLANT_TEMP','AIR_PRESSURE_DIGITAL','NO_OF_SAT','PACKET_STATUS','LOCATION', \
# 	'B.PACKET_TYPE','LOCAL_TIME_STAMP','LATITUDE','LONGITUDE','WHEEL_SPEED','RESERVE1','RESERVE2','KAFKA_TIME_STAMP','RECEIVED_TIMESTAMP','VERSION','IGNITION_STATUS','VEH_STATUS','INTERNAL_BATTERY_POTENTIAL','PACKET_CODE','CAL_ODO','VEHICLE_ODO'],axis=1,inplace=True)
data.drop(data[['EFF_TORQUE','OBU_ID','B.PACKET_TYPE','EVENT_UTC','LOCAL_TIME_STAMP','LATITUDE','LONGITUDE','RESERVE1','RESERVE2','KAFKA_TIME_STAMP','RECEIVED_TIMESTAMP','VERSION']],axis=1,inplace=True)
# data = data.dropna(subset=['GPS_SPEED','ENGINE_SPEED'])
filt= (data['ENGINE_SPEED']!=0) & (data['GPS_SPEED'] !=0)
data = data[filt]
filt_1 = (data['ENGINE_SPEED']>700)
data = data[filt_1]

data = data.dropna()
print(data.info())
print(data.head())
print(data.shape) 

# sb.heatmap(data.corr(),annot=True,fmt='.1f')
# plt.show()
# plt.figure(figsize=(10,7))
# plt.scatter(data['GPS_SPEED'],data['ENGINE_SPEED'])
# # plt.scatter(data['YELLOW'],data['KMPL'],c='yellow')
# plt.show()



# df_train = data[:42500]    # 80% training data and 20% testing data
# df_test = data[42500:]
# scaler = MinMaxScaler() # For normalizing dataset
# # We want to predict Close value of stock 
# X_train = scaler.fit_transform(df_train.drop(['ENGINE_SPEED'],axis=1).values.reshape(-1,1))
# y_train = scaler.fit_transform(df_train['ENGINE_SPEED'].values.reshape(-1,1))
# # y is output and x is features.
# X_test = scaler.fit_transform(df_test.drop(['ENGINE_SPEED'],axis=1).values.reshape(-1,1))
# y_test = scaler.fit_transform(df_test['ENGINE_SPEED'].values.reshape(-1,1))

# def denormalize(df,norm_data):
#     data = data['ENGINE_SPEED'].values.reshape(-1,1)
#     norm_data = norm_data.reshape(-1,1)
#     scl = MinMaxScaler()
#     a = scl.fit_transform(data)
#     new = scl.inverse_transform(norm_data)



# n_features = X_train.shape[0]
# print(n_features,df_train.shape)

# n_neurons_1 = 500
# n_neurons_2 = 500
# n_neurons_3 = 500


# X = tf.placeholder(dtype=tf.float32, shape=[None,n_features])
# Y = tf.placeholder(dtype=tf.float32,shape=[None,1])

# sigma = 1
# weight_initializer = tf.variance_scaling_initializer(mode='fan_avg',distribution = 'uniform',scale=sigma)
# bias_initializer = tf.zeros_initializer()


# w_hidden_1 = tf.Variable(weight_initializer([n_features, n_neurons_1]))
# bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))


# w_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
# bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

# w_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
# bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))


# w_out = tf.Variable(weight_initializer([n_neurons_3, n_features]))
# bias_out = tf.Variable(bias_initializer([n_features]))


# hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, w_hidden_1),bias_hidden_1))
# hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, w_hidden_2),bias_hidden_2))
# hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, w_hidden_3),bias_hidden_3))

# out = tf.transpose(tf.add(tf.matmul(hidden_3,w_out),bias_out)) #why transpose

# mse = tf.reduce_mean(tf.squared_difference(out,Y))
# #mse = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
# opt = tf.train.GradientDescentOptimizer(0.001).minimize(mse)

# batch_size = 100
# epochs = 10


# # Add an op to initialize the variables.
# init_op = tf.global_variables_initializer()
# saver = tf.train.Saver()
# #init_lo = tf.local_variables_initializer()
# with tf.Session() as sess:
# 	sess.run(init_op)#initialize_all_variables()
# 	saver.restore(sess,'E:/Eduvance/machine-learning-course/nn_1.ckpt')

# 	for e in range(epochs):
# 		c_t=[]
# 		c_test=[]
# 		epoch_loss = 0
# 		#shuffle_data = np.random.permutation(np.arange(len(y_train)))
# 		#X_train = X_train[shuffle_data]
# 		#y_train = y_train[shuffle_data]
# 		for i in range(0, len(y_train)//batch_size):
# 			start = i*batch_size
# 			batch_x = X_train[start:start + batch_size]
# 			batch_y = y_train[start:start + batch_size]
# 			_,c = sess.run([opt,mse], feed_dict = {X:batch_x, Y:batch_y})
# 			epoch_loss += c
			
# 		print('Epoch', e,'completed out of', epochs,'loss:', epoch_loss)
# # 	saver.save(sess, 'E:/Eduvance/machine-learning-course/nn_1.ckpt')

# 		c_t.append(sess.run(mse, feed_dict={X:X_train,Y:y_train}))
# 		c_test.append(sess.run(mse, feed_dict={X:X_test,Y:y_test}))
# 		print('Epoch :',e,'Cost :',c_t[e])
# 		# predict output of test data after training
# 	pred = sess.run(out, feed_dict={X:X_test})
# 	print('Cost :',sess.run(mse, feed_dict={X:X_test,Y:y_test}))
# 	print(pred)
# 	print(c_t,c_test)












