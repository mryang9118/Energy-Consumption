"""
@Project: Energy-Consumption   
@Description: calculate the feature importance, then decide the best configuration
@Time:2020/11/4 17:31
"""
from sklearn.pipeline import Pipeline
from ml_models import ModelsFitter
from utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


warnings.filterwarnings(action="ignore")
data_path = "../data/Volkswagen_2000.csv"
after_clean = clean_ev_data(data_path)
X, y = preprocess_data(after_clean, SPRIT_MONITOR_X_COLUMN_NAMES,
                       SPRIT_MONITOR_TARGET_COLUMN_NAME, SPRIT_MONITOR_REQUIRE_ENCODED_COLUMNS)
X_train_original, X_test, y_train, y_test = train_test_split(X[1], y, test_size=0.4, shuffle=True)

r2_score_list = []
options_feature_count = range(1, 13)
for i in options_feature_count:
    X_train = X_train_original
    print('-----------Select %s features-----------' % i)
    model_fitter = ModelsFitter(RF)
    pipeline = Pipeline([
         (DECOMPOSER, PCA(n_components=i)),
         (STANDARD_SCALAR, StandardScaler()),
         (ESTIMATOR, model_fitter)
     ])
    pipeline.fit(X_train, y_train)
    scores_result = model_fitter.evaluate_model()
    r2_score_list.append(scores_result['r2'])

plt.title('Evaluation For Feature Extraction')
plt.grid()
plt.plot(options_feature_count, r2_score_list, linewidth=2, marker='o', markersize=7, color="b")
plt.xlabel('Feature Counts')
plt.ylabel('Score')
plt.show()