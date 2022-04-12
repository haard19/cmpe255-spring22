import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, confusion_matrix, plot_confusion_matrix, accuracy_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
np.random.seed(0)
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("fetal_health.csv")
# data.head()
# data.describe()

# TODO: 1. Check the distribution [Tagged as 1 (Normal), 2 (Suspect) and 3 (Pathological)] of target label 'fetal_health' using a SNS Countplot
#

# TODO: 2. Find the top 3 input features based on the correlation matrix by plotting a SNS heatmap.
#

X = data.drop(['fetal_health'], axis=1)
y = data["fetal_health"]

train_size, num_features = X.shape
num_labels = 3

# TODO: 3. Use Scikit-Learn Standard Scaler to normalize the input features.


# Split the dataset into a training set (80%) and a test set (20%) using train_test_split() from sklearn library.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=109)

# TODO: 4. Use Scikit-Learn's LogisticRegression 
# set these three parameters to enable the Softmax Regression algorithm:
#   multi_class="multinomial"
#   solver="lbfgs"
#   C=10
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

print('| Experiment | Accuracy | Recall | Precision | Confusion matrix |')
print('|------------|----------|----------|----------|---------|')

for i in [6,7,8]:
    clf = SVC(C=i, kernel="linear")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # print(y_test, y_pred)
    # TODO: 5. Compute these scores
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    cf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(clf, X_test, y_test)
    plt.show()

    print(f"|Solution{i-5}   | {accuracy} | {recall} | {precision} | {cf_matrix} |")
    # print("Accuracy    : ", accuracy)
    # print("Recall      : ", recall)
    # print("Precision   : ", precision)
    # print("Confusion Matrix: ", cf_matrix)