from sklearn.preprocessing import StandardScaler
import numpy as np
data = np.random.randn(10, 4)
scaler = StandardScaler()
scaler.fit(data)
trans_data = scaler.transform(data)
print('original data: ')
print(data)
print('transformed data: ')
print(trans_data)
print('scaler info: scaler.mean_: {}, scaler.var_: {}'.format(scaler.mean_, scaler.var_))
print('\n')