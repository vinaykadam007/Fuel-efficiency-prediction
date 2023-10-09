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
data.drop(data[['THROTTLE_POS','WHEEL_SPEED','EFF_TORQUE','OBU_ID','B.PACKET_TYPE','EVENT_UTC','LOCAL_TIME_STAMP','LATITUDE','LONGITUDE','RESERVE1','RESERVE2','KAFKA_TIME_STAMP','RECEIVED_TIMESTAMP','VERSION']],axis=1,inplace=True)
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



df_train = data[:48500]    # 80% training data and 20% testing data
df_test = data[48500:]
scaler = MinMaxScaler() # For normalizing dataset
# We want to predict Close value of stock 
X_train = scaler.fit_transform(df_train.drop(['ENGINE_SPEED'],axis=1).values)
y_train = scaler.fit_transform(df_train['ENGINE_SPEED'].values.reshape(-1,1))
# y is output and x is features.
X_test = scaler.fit_transform(df_test.drop(['ENGINE_SPEED'],axis=1).values)
y_test = scaler.fit_transform(df_test['ENGINE_SPEED'].values.reshape(-1,1))
print(X_train.shape)
#print(X_train.reshape(-1,3))



def neural_net_model(X_data,input_dim):
    W_1 = tf.Variable(tf.random_uniform([input_dim,10]))
    b_1 = tf.Variable(tf.zeros([10]))
    layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)

    # layer 1 multiplying and adding bias then activation function
    W_2 = tf.Variable(tf.random_uniform([10,10]))
    b_2 = tf.Variable(tf.zeros([10]))
    layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
    layer_2 = tf.nn.relu(layer_2)
    # layer 2 multiplying and adding bias then activation function
    W_3 = tf.Variable(tf.random_uniform([10,10]))
    b_3 = tf.Variable(tf.zeros([10]))
    layer_3 = tf.add(tf.matmul(layer_2,W_3), b_3)
    layer_3 = tf.nn.relu(layer_3)
    # layer 3 multiplying and adding bias then activation function
    W_O = tf.Variable(tf.random_uniform([10,1]))
    b_O = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(layer_2,W_O), b_O)
    # O/p layer multiplying and adding bias then activation function
    # notice output layer has one node only since performing #regression
    return output

xs = tf.placeholder("float")
ys = tf.placeholder("float")
output = neural_net_model(xs,1)
cost = tf.reduce_mean(tf.square(output-ys))
# our mean squared error cost function
train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
# Gradinent Descent optimiztion just discussed above for updating weights and biases

# batch_size = 100
# epochs = 10


with tf.Session() as sess:
    # Initiate session and initialize all vaiables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # saver.restore(sess,'C:/Users/Anil/Desktop/nn_1.ckpt')
    c_test=[]
    c_t=[]
    for i in range(10):
        for j in range(0, X_train.shape[0]):
        	# start = i*batch_size
        	# batch_x = X_train[start:start + batch_size]
        	# batch_y = y_train[start:start + batch_size]
        	sess.run([cost,train],feed_dict= {xs:X_train[j,:].reshape(-1,1) ,ys:y_train[j]})
            # Run cost and train with each sample
        c_t.append(sess.run(cost, feed_dict={xs:X_train,ys:y_train}))
        c_test.append(sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
        print('Epoch :',i,'Cost :',c_t[i])
    # saver.save(sess, 'C:/Users/Anil/Desktop/nn_1.ckpt')
    pred = sess.run(output, feed_dict={xs:X_test})
    # predict output of test data after training
    print('Cost :',sess.run(cost, feed_dict={xs:X_test,ys:y_test}))

    print(c_test)
    print(pred)
    data = data['ENGINE_SPEED'].values.reshape(-1,1)
    pred = pred.reshape(-1,1)
    scl = MinMaxScaler()
    a = scl.fit_transform(data)
    predicted = scl.inverse_transform(pred)
    print(predicted)


    y_test = y_test.reshape(-1,1)
    scl_1 = MinMaxScaler()
    a_1 = scl_1.fit_transform(data)
    test = scl_1.inverse_transform(y_test)
    print(test)
plt.figure()
plt.plot(range(y_test.shape[0]),test,label="Original Data")
plt.plot(range(y_test.shape[0]),predicted,label="Predicted Data")
plt.legend()
plt.show()

















































































