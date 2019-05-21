#import needed packages
from sklearn.datasets import fetch_mldata
#from sklearn import metrics
from sklearn.model_selection import train_test_split
#import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#fetch data
mnist = mnist = fetch_mldata('MNIST original', transpose_data=True, data_home='files')

#data should appear as follows:
'''
{'COL_NAMES': ['label', 'data'],
 'DESCR': 'mldata.org dataset: mnist-original',
 'data': array([[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ...,
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]], dtype=uint8),
 'target': array([0., 0., 0., ..., 9., 9., 9.])}
'''

#confirm dimension of array - should be (70000, 786)
print("Total data set size: ", mnist.data.shape)

#confirm labels array is present - should be (70000,)
print("Total label data set size: ", mnist.target.shape)

#split array into training and testing arrays
train_img, test_img, train_lbl, test_lbl = train_test_split(
    mnist.data, mnist.target, test_size=1/7.0, random_state=0)

#confirm dimensionality of arrays - should be (60000) in training and (10000) in testing
print("Training set size: ", train_img.shape)
print("Training label set size: ", train_lbl.shape)
print("Testing set size: ", test_img.shape)
print("Testing label set size: ", test_lbl.shape)

#standardize data using StandardScaler - this converts data into norm with mean = 0 and SD = 1
#https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(train_img)

# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

#import PCA
from sklearn.decomposition import PCA
pca = PCA(.95)

#fit the training data
pca.fit(train_img)
#print(pca.explained_variance_ratio_)

#map both data sets
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

print("Training image shape: ", train_img.shape)
print("Test image shape: ", test_img.shape)

#plot the resulting transform
# plt.scatter(test_img[:, 0], test_img[:, 1],
#             c=test_lbl, edgecolor='none', alpha=0.5,
#             cmap=plt.cm.get_cmap('Spectral', 10))
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.colorbar();
# plt.show()

#3d graph?
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate the values
x_vals = test_img[0:1000, 0]
y_vals = test_img[0:1000:, 1]
z_vals = test_img[0:1000:, 2]

# Plot the values
ax.scatter(x_vals, y_vals, z_vals, c=test_lbl[0:1000], edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('Spectral', 10))
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()

#now run the machine learning regression
from sklearn.linear_model import LogisticRegression

# all parameters not specified are set to their defaults
# default solver is incredibly slow thats why we change it
# solver = 'lbfgs', set verbose=True to get timing and output
logisticRegr = LogisticRegression(solver = 'lbfgs')

logisticRegr.fit(train_img, train_lbl)

# Returns a NumPy Array
# Predict for One Observation (image)
logisticRegr.predict(test_img[0].reshape(1,-1))

# Predict for Multiple Observations (images) at Once
logisticRegr.predict(test_img[0:10])

#score accuracy against known labels
score = logisticRegr.score(test_img, test_lbl)
print(score)