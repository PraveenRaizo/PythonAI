import numpy as np 
import matplotlib.pyplot as plt 
import neurolab as nl 

#generating some datapoint based on this equation : y = 2x^2 + 8

min_val = -30
max_val = 30
num_points = 160
x = np.linspace(min_val, max_val, num_points) # generates equally spaced 160 numbers between -30 to 30 
print("generated x = ", x,"\n")
y = 2*np.square(x) + 8
print("generated y based on 2x^2 + 8 formula: ", y)
y /= np.linalg.norm(y) # this does the L2/Euclidean normalisation of the data in y. Euclidean norm of a vector is the sqroot of the sum of the squares of its elements
print("generated y after normalisation: ")

# now we are going to reshape the data
data = x.reshape(num_points, 1)
labels = y.reshape(num_points, 1)

# we are now visualizing and plotting the input dataset
plt.figure()
plt.scatter(data, labels)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Data-Points')

#now we are gonna build a neural network having two hidden layers with neurolab with ten neurons in the first hidden layer, 6 in second hidden layer, and on in output layer
neural_net = nl.net.newff([[min_val, max_val]], [10,6,1])

#now use gradient training algorithm
neural_net.trainf = nl.train.train_gd

#now train the network with goal of learning on the data generated above
error = neural_net.train(data, labels, epochs = 1000, show =100, goal=0.01)

# now run the neural networks on the training data-points
output = neural_net.sim(data)
y_pred = output.reshape(num_points)

#now plot and visualization task
plt.figure()
plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')

#now we will be plotting the actual versus predicted output
x_dense = np.linspace(min_val, max_val, num_points*2)
y_dense_pred = neural_net.sim(x_dense.reshape(x_dense.size, 1)).reshape(x_dense.size)
plt.figure()
plt.plot(x_dense, y_dense_pred, '-', x, y, '.', x, y_pred, 'p')
plt.title('Actual vs Predicted')
plt.show()







