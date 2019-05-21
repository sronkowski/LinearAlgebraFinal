#import needed packages
from sklearn.datasets import fetch_mldata

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd

#fetch data
mnist = mnist = fetch_mldata('MNIST original', transpose_data=True, data_home='files')

#confirm data is present - output should be:
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
mnist

#confirm dimension of array - should be (70000, 786)
mnist.data.shape

#confirm labels array is present - should be (70000,)
mnist.target.shape

#split array into training and testing arrays
train_img, test_img, train_lbl, test_lbl = train_test_split(
    mnist.data, mnist.target, test_size=1/7.0, random_state=0)

#confirm dimensionality of arrays - should be (60000) in training and (10000) in testing
print(train_img.shape)
print(train_lbl.shape)
print(test_img.shape)
print(test_lbl.shape)



