# load and evaluate a saved model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import load_model

# load model
model = load_model('./volkswagen_e_golf_model.h5')
# summarize model.
model.summary()
# load dataset
dataset = pd.read_csv(filepath_or_buffer = '../data/volkswagen_e_golf_data.csv')
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
X_test = sc.fit_transform(X=X_test)
test_pred = model.predict(X_test)
print("RMSE on test data: %.3f" % np.sqrt(mean_squared_error(y_true=y_test, y_pred=test_pred)))
print("MAE on test data: %.3f" % mean_absolute_error(y_true=y_test, y_pred=test_pred))
print("variance score on test data: %.3f" % r2_score(y_true=y_test, y_pred=test_pred))
