import numpy as np
from sklearn import preprocessing

Input_data = np.array([[2.1, -1.9, 5.5],
                      [-1.5, 2.4,  3.5],
                      [0.5, -7.9, 5.6],
                      [5.9,2.3,-5.8]])

# binarization: 

data_binarized = preprocessing.Binarizer(threshold=0.5).transform(Input_data)
print("\nBinarized data:\n",data_binarized)

#getting the mean and sd
print("Mean=", Input_data.mean(axis = 0))
print("Std deviation = ", Input_data.std(axis=0))

#removing the mean and sd from Input data to get a scaled data
data_scaled = preprocessing.scale(Input_data)
print("Mean = ", data_scaled.mean(axis=0))
print("Std deviation = ", data_scaled.std(axis =0))

#scaling the feature vectors - a data preprocessing technique that is used to scale the feature vectors
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scaler_minmax.fit_transform(Input_data)
print("\nMin Max scaled data:\n", data_scaled_minmax)

#L1 Normalization: Least Absolute Deviations  (here we normalize the data)
#this kind of normalization modifies the values so that the sum of the absolute values in always up to 1 in each row
data_normalized_l1 = preprocessing.normalize(Input_data, norm='l1')
print("\n L1 normalized data:\n", data_normalized_l1)

#Normalize data: l2 
#this is also referred to as least squares. 
#this kind of normalizaion modifies the values so that the sum of the squares is always up to 1 in each row

data_normalized_l2 = preprocessing.normalize(Input_data, norm='l2')
print("\n L2 normalized data:\n", data_normalized_l2)


###################################################################

# Sample input labels
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']

# Creating the label encoder
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

#LabelEncoder()

#encoding a set of labels
test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)
print("\n Labels = ", test_labels)

Labels = ['green', 'red', 'black']
print("encoded values =", list(encoded_values))

# decoding a set of values:

encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values = ", encoded_values)
print("\nDecoded values = ", decoded_list)













