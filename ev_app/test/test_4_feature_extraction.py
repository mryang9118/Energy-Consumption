"""
@Project: Energy-Consumption   
@Description: calculate the feature importance, then decide the best configuration
@Time:2020/11/4 17:31
"""
from ml_models.preprocess import *
from utils.constants import *
from sprit_monitor.sprit_monitor_preprocess import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
from sklearn.decomposition import PCA
from ml_models.models_getter import *
import matplotlib.pyplot as plt


warnings.filterwarnings(action="ignore")
data_path = "../data/Volkswagen_2000.csv"
after_clean = clean_ev_data(data_path)
X, y = preprocess_data(after_clean, X_COLUMN_NAMES, Y_COLUMN_NAME, REQUIRE_ENCODED_COLUMNS)
X_train_original, X_test, y_train, y_test = train_test_split(X[1], y, test_size=0.4, shuffle=True)

r2_score_list = []
options_feature_count = range(1, 12)
for i in options_feature_count:
    X_train = X_train_original
    print('-----------Select %s features-----------' % i)
    pca = PCA(n_components=i)
    # prepare transform on dataset
    X_train = pca.fit_transform(X_train)
    sc = StandardScaler()
    X_train = sc.fit_transform(X=X_train)
    model_fitter = ModelsFitter(RF, X_train, y_train)
    model_fitter.fit_model()
    scores_result = model_fitter.evaluate_model()
    r2_score_list.append(scores_result['r2'])

plt.title('Evaluation For Feature Extraction')
plt.grid()
plt.plot(options_feature_count, r2_score_list, linewidth=2, marker='o', markersize=7, color="b")
plt.xlabel('Feature Counts')
plt.ylabel('Score')
plt.show()