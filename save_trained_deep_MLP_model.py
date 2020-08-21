import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(units=100, kernel_initializer='uniform', activation='relu', input_dim=len(X[0])))
    regressor.add(Dense(units=50, kernel_initializer='uniform', activation='relu'))
    regressor.add(Dense(units=25, kernel_initializer='uniform', activation='relu'))
    regressor.add(Dense(units=13, kernel_initializer='uniform', activation='relu'))
    regressor.add(Dense(units=7, kernel_initializer='uniform', activation='relu'))
    # activation func of the output layer must be 'linear' for regression tasks
    regressor.add(Dense(units=1, kernel_initializer='uniform', activation='linear'))
    regressor.compile(optimizer='adam', loss='mean_absolute_error')
    return regressor

new_path = "./volkswagen_e_golf_data.csv"
dataset = pd.read_csv(filepath_or_buffer=new_path)
filter_condition = np.abs(dataset['quantity(kWh)']/dataset['trip_distance(km)'] * 100 - dataset['consumption(kWh/100km)']) < dataset['consumption(kWh/100km)']/2
dataset = dataset[filter_condition]
X = dataset.iloc[:, 4:15].values
y = dataset.iloc[:, 3].values


"""do the preprocessing tasks on the data"""
# encode categorical features
label_encoder_1 = LabelEncoder()
X[:, 0] = label_encoder_1.fit_transform(y=X[:, 0])
label_encoder_1 = LabelEncoder()
X[:, 2] = label_encoder_1.fit_transform(y=X[:, 2])
label_encoder_2 = LabelEncoder()
X[:, 6] = label_encoder_2.fit_transform(y=X[:, 6])

# onehot encoding for categorical features with more than 2 categories
onehot_encoder = OneHotEncoder(categorical_features=[0, 2, 6])
X = onehot_encoder.fit_transform(X=X).toarray()

# delete the first column to avoid the dummy variable
X = np.delete(X, [0, 4, 7], 1)

# split the dataset into training-set and test-set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# scale the values
sc = StandardScaler()
X_train = sc.fit_transform(X=X_train)
# X_test = sc.fit_transform(X=X_test)
deep_mlp = KerasRegressor(build_fn=build_regressor, batch_size=8, epochs=50, verbose=False)
deep_mlp.fit(X_train, y_train)
deep_mlp.model.save("./volkswagen_e_golf_model.h5")
print("Saved model to disk")