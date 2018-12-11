import numpy as np

#
data = np.genfromtxt('opt/test/breast_cancer.csv', delimiter=',')
feature_values = data[:, :-1]

zero_mean_unit_var_features = (feature_values - np.mean(feature_values, 0)) / np.std(feature_values, 0)
data[:, :-1] = zero_mean_unit_var_features

np.savetxt('opt/test/breast_cancer_zero_mean_unit_var.csv', data, delimiter=",", fmt='%f')

# test = np.array([
#     [1, 2],
#     [4, 5],
#     [7, 8]
# ])
# print np.mean(test, 0)
# print np.std(test, 0)
#
# print (test - np.mean(test, 0))/ np.std(test, 0)
