import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def do_kfold(model):
    cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=41)
    acc_scores = cross_val_score(estimator=model, X=X, y=y, scoring='balanced_accuracy', cv=cv)
    print('accuracy scores:', acc_scores)
    print("average accuracy score (bias) is:", abs(round(number=acc_scores.mean() * 100, ndigits=3)))
    print("std deviation of MAE scores (variance) is:", round(number=acc_scores.std() * 100, ndigits=3))
    best_acc = sorted(acc_scores, reverse=False)[-1]
    print("best accuracy score is:", abs(round(number=best_acc * 100, ndigits=3)))
    print("-------------------------------")


#
def do_train_test(model):
    model.fit(X_train, y_train)
    training_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    training_acc = accuracy_score(y_true=y_train, y_pred=training_pred)
    test_acc = accuracy_score(y_true=y_test, y_pred=test_pred)
    print("accuracy on training-set:", training_acc)
    print("accuracy on test-set:", test_acc)
    cm = confusion_matrix(y_true=y_test, y_pred=test_pred)
    print("confusion matrix on test-set:", cm)
    print("-------------------------------")


warnings.filterwarnings(action="ignore")
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

old_path = "./data/volkswagen_e_golf_clean.csv"
new_path = "./data/new_volkswagen_e_golf.csv"


"""remove missing values (comment it after the first run)"""
# ds = pd.read_csv(filepath_or_buffer=old_path)
# ds = ds[pd.notnull(obj=ds['quantity(kWh)'])]
# ds = ds[pd.notnull(obj=ds['avg_speed(km/h)'])]
# ds = ds[pd.notnull(obj=ds['consumption(kWh/100km)'])]
# ds.to_csv(path_or_buf=new_path)


"""load the data"""
dataset = pd.read_csv(filepath_or_buffer=new_path)
# print(dataset.head(n=5))
# print(dataset.describe())

X = dataset.iloc[:, 4:14].values
y = dataset.iloc[:, 14].values
# consumption_values = dataset.iloc[:, 16].values


"""change ECR deviation values into binary values:
if real ECR is more than manufacture pre-defined ECR -> put 1
if real ECR is less than manufacture pre-defined ECR -> put 0"""
y = (y >= 0)
y = np.array(y, dtype='int')


"""do the preprocessing tasks on the data"""
# encode categorical features
label_encoder_1 = LabelEncoder()
X[:, 2] = label_encoder_1.fit_transform(y=X[:, 2])
label_encoder_2 = LabelEncoder()
X[:, 6] = label_encoder_2.fit_transform(y=X[:, 6])

# onehot encoding for categorical features with more than 2 categories
onehot_encoder = OneHotEncoder(categorical_features=[6])
X = onehot_encoder.fit_transform(X=X).toarray()

# delete the first column to avoid the dummy variable
X = X[:, 1:]

# split the dataset into training-set and test-set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X=X_train)
X_test = sc.fit_transform(X=X_test)


"""find the best parameters for the SVM classifier using GridSearch (comment it after the first run)"""
parameters = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'), 'C': [0.1, 1, 10, 100], 'degree': [1, 2, 3, 4, 5]}
svm_classifier = svm.SVC()
clf = GridSearchCV(estimator=svm_classifier, param_grid=parameters)
clf.fit(X=X_train, y=y_train)
print(clf.best_params_)


"""define the SVM classification model with best parameters obtained from above"""
svm_classifier = svm.SVC(C=1000, kernel='rbf', gamma=0.12)

"""KFold cross-validation"""
print("\n ------ SVM CrossVal ------")
do_kfold(model=svm_classifier)

"""evaluate on the never-seen-before test data"""
print("\n ------ SVM TrainTest ------")
do_train_test(model=svm_classifier)


"""define the MLP as the classifier"""
mlp_classifier = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam',
                               verbose=False, max_iter=1000, n_iter_no_change=100, warm_start=False)

print("\n ------ MLP Classifier CrossVal ------")
do_kfold(model=mlp_classifier)

print("\n ------ MLP Classifier TrainTest ------")
do_train_test(model=mlp_classifier)


"""define the RF as the classifier"""
rf_classifier = RandomForestClassifier(n_estimators=50)

print("\n ------ Random Forest Classifier CrossVal ------")
do_kfold(model=rf_classifier)

print("\n ------ Random Forest Classifier TrainTest ------")
do_train_test(model=rf_classifier)


"""define the deep mlp as the classifier"""
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=100, kernel_initializer='uniform', activation='relu', input_dim=len(X[0])))
    classifier.add(Dense(units=50, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=25, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=13, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=7, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


deep_mlp_classifier = KerasClassifier(build_fn=build_classifier, batch_size=16, epochs=1000, verbose=False)

print("\n ------ Deep MLP Classifier CrossVal ------")
do_kfold(model=deep_mlp_classifier)

print("\n ------ Deep MLP Classifier TrainTest ------")
do_train_test(model=deep_mlp_classifier)