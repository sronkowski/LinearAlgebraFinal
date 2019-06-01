#import needed packages
from sklearn.datasets import fetch_mldata
#from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import timeit
from sklearn.linear_model import LogisticRegression

#fetch data
mnist = fetch_mldata('MNIST original', transpose_data=True, data_home='files')

#confirm dimension of array - should be (70000, 784)
#print("Total data set size: ", mnist.data.shape)
#confirm labels array is present - should be (70000,)
# print("Total label data set size: ", mnist.target.shape)

#split array into training and testing arrays
train_img, test_img, train_lbl, test_lbl = train_test_split(
    mnist.data, mnist.target, test_size=1/7.0, random_state=0)

#image output of a digit
#https://maxpowerwastaken.github.io/blog/exploring-the-mnist-digits-dataset/
def showTransformedNumber(pca, train_lbl, train_img):
    y = pd.Series(train_lbl).astype('int').astype('category')
    X = pd.DataFrame(pca.inverse_transform(train_img[:,0:train_img.shape[1]]))
    num_images = X.shape[1]
    X.columns = ['pixel_'+str(x) for x in range(num_images)]
    # First row is first image
    first_image = X.loc[0,:]
    first_label = y[0]
    plottable_image = np.reshape(first_image.values, (28, 28))
    # Plot the image
    plt.imshow(plottable_image, cmap='gray_r')
    plt.title('Digit Label: {}'.format(first_label))
    plt.show()

def showNumber(train_lbl, train_img):
    y = pd.Series(train_lbl).astype('int').astype('category')
    X = pd.DataFrame(train_img[:,0:train_img.shape[1]])
    num_images = X.shape[1]
    X.columns = ['pixel_'+str(x) for x in range(num_images)]
    # First row is first image
    first_image = X.loc[0,:]
    first_label = y[0]
    # 784 columns correspond to 28x28 image
    plottable_image = np.reshape(first_image.values, (28, 28))
    # Plot the image
    plt.imshow(plottable_image, cmap='gray_r')
    plt.title('Digit Label: {}'.format(first_label))
    plt.show()

#confirm dimensionality of arrays - should be (60000) in training and (10000) in testing
# print("Training set size: ", train_img.shape)
# print("Training label set size: ", train_lbl.shape)
# print("Testing set size: ", test_img.shape)
# print("Testing label set size: ", test_lbl.shape)

#original image display
showNumber(train_lbl, train_img)

#standardize data using StandardScaler - this converts data into norm with mean = 0 and SD = 1
#https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(train_img)

# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

#normalized image display
showNumber(train_lbl, train_img)

def pcaIterate(nc=324, train_img=train_img, test_img=test_img, train_lbl=train_lbl):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=nc, whiten=True)
    # fit the training data
    pca.fit(train_img)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    # map both data sets
    train_img = pca.transform(train_img)
    test_img = pca.transform(test_img)
    print("Training image shape: ", train_img.shape)
    print("Test image shape: ", test_img.shape)
    showTransformedNumber(pca, train_lbl, train_img)
    return train_img, test_img

train_img, test_img = pcaIterate()

#plot the resulting transform
# plt.scatter(test_img[0:6000, 0], test_img[0:6000, 1],
#              c=test_lbl[0:6000], edgecolor='none', alpha=0.5,
#              cmap=plt.cm.get_cmap('Spectral', 10))
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.colorbar();
# plt.show()

#3d graph?
# ax = plt.axes(projection='3d')
# # Generate the values
# x_vals =test_img[:, 0]
# y_vals = test_img[:,1]
# z_vals = test_img[:, 2]
#
# # Plot the values
# ax.scatter(x_vals, y_vals, z_vals, c=test_lbl[:], cmap=plt.cm.get_cmap('Spectral', 10))
# plt.show()



#now run the machine learning regression

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
print("Accuracy Score:", score)