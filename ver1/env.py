import numpy as np
from numpy.core.arrayprint import format_float_scientific
import pandas as pd
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer, normalize

class User:
    def __init__(self, feature_num, dataset_num):
        self.feature_num = feature_num
        self.dataset_num = dataset_num
        
    def generate_user_normal(self, mu, sigma):
        x = np.random.normal(mu, sigma, self.feature_num)
        return x
    
    def generate_user_normal_set(self, dataset_num, mu, sigma):
        USERS = []
        for i in range(dataset_num):
            USERS.append(self.generate_user_normal(mu, sigma))
        return np.array(USERS)
    
    ###
    def genitems(self, mu, sigma):
        A = np.random.normal(mu, sigma, (self.dataset_num, self.feature_num-1))    # A: (L, d-1)
        result = np.hstack((normalize(A, axis=1) / np.sqrt(2), np.ones((self.dataset_num, 1)) / np.sqrt(2)))   # result: (L, d)
        return result
    
    def __str__(self):
        return '\n'
    

class Theta:
    def __init__(self, shape):
        self.shape = shape
    
    def true_normal(self, mu, sigma):
        true_normal = np.random.normal(mu, sigma, self.shape)
        return true_normal
    
    def __str__(self):
        return '\n'


class Y:
    def __init__(self):
        self.Y = []
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    # def normalizer(self, Y):
    #     normal = Normalizer()
    #     normalized_Y = normal.fit(Y)
    #     return normalized_Y
        
    def true_mean_linear(self, X, true_theta):
        X_binary = []
        mean = np.matmul(X, true_theta)
        return mean
    
    def genitems(self, dataset_num, feature_num):
        A = np.random.normal(0, 1, (dataset_num, feature_num-1)) # A: (L, d-1)
        result = np.hstack((normalize(A, axis=1) / np.sqrt(2), np.ones((dataset_num, 1)) / np.sqrt(2))) # result: (L, d)
        return result
    
    def normalized_true_mean_linear(self, dataset_num, feature_num):
        items = self.normalizer(dataset_num, feature_num)
        theta = self.normalizer(1, feature_num)[0]
        means = np.dot(items, theta)
        return means
    
    # def 
        
    def normalizer(self, mean):
        for mean_val in mean:
            mean_sigmoid = self.sigmoid(mean_val)
            X_binary = np.where(mean_sigmoid>0.5, 1,0)
            self.Y.append(X_binary)         
        # return np.array(self.Y)
        return self.Y
    
    def __str__(self):
        return '\n'
        


# ###
#     def genitems(self, L, d):
#         A = np.random.normal(0, 1, (L, d-1))    # A: (L, d-1)
#         result = np.hstack((normalize(A, axis=1) / np.sqrt(2), np.ones((L, 1)) / np.sqrt(2)))   # result: (L, d)
#         return result

#     self.items = self.genitems(L, d)        # items: (L, d)
#     theta = self.genitems(1, d)[0]          # theta: (d,)
#     self.means = np.dot(self.items, theta)  # means: (L,)

