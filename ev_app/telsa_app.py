"""
@Project: Energy-Consumption   
@Description: telsa application
@Time:2020/10/14 15:41                      
 
"""
import warnings
from sprit_monitor.sprit_monitor_preprocess import *
from sklearn.preprocessing import StandardScaler
from ml_models.preprocess import *
from utils.constants import *
from ml_models.parse_models import ModelsGetter
from ml_models.evaluate_util import *
from sklearn.model_selection import train_test_split

warnings.filterwarnings(action="ignore")
file_path = "../data/tesla_trace.csv"
SOURCE_COLUMN_NAMES = ['battery_range', 'speed', 'temperature', 'temperature_setting', 'battery_level']
TARGET_COLUMN_NAMES = ['odometer']
# preprocess ev data from sprit monitor
raw_data = pd.read_csv(filepath_or_buffer=file_path, delimiter=' ')
X, y = preprocess_data(raw_data, SOURCE_COLUMN_NAMES, TARGET_COLUMN_NAMES, [])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True)

# scale the values
sc = StandardScaler()
X_train = sc.fit_transform(X=X_train)
X_test = sc.fit_transform(X=X_test)
# RF Model
print('-------RF Start-------')
getter = ModelsGetter(RF, X_train, y_train)
getter.process()
model = getter.get_model()
y_pred = model.predict(X_test)
print('y test: %s' % str(y_test))
print('y prediction: %s' % str(y_pred))
evaluate_predict_result(y_test, y_pred)
print('-------RF End-------')
# Deep MLP Model
print('-------Deep MLP Start-------')
getter = ModelsGetter(DEEP_MLP, X_train, y_train)
getter.process()
model = getter.get_model()
y_pred = model.predict(X_test)
evaluate_predict_result(y_test, y_pred)
print('-------Deep MLP End-------')