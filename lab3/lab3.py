import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_selection import RFE

        
class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
        # print(self.pima.head())
        self.X_test = None
        self.y_test = None


    def train(self,x):
        # split X and y into training and testing sets
        X, y = self.feature_def(x)
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, random_state=12345)
        # train a logistic regression model on the training set
        logreg = LogisticRegression(solver='lbfgs', max_iter=10000)
        logreg.fit(X_train, y_train)
        return logreg
    

    def predict(self, x):
        model = self.train(x)
        y_pred_class = model.predict(self.X_test)
        return y_pred_class


    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)


    def examine(self):
        dist = self.y_test.value_counts()
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()
    

    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)


    def addBP(self):
        colBP = pd.Series(
            ["low", "high"], dtype="category")
        self.pima["colBP"] = colBP

        self.pima.loc[self.pima["bp"] < 50, "colBP"] = colBP[0]
        self.pima.loc[self.pima["bp"] > 50, "colBP"] = colBP[1]


    def addGlucose(self):
        colGlucose = pd.Series(
            ["normal", "prediabetes", "diabetic"], dtype="category")
        self.pima["colGlucose"] = colGlucose

        self.pima.loc[self.pima["glucose"] < 140, "colGlucose"] = colGlucose[0]
        self.pima.loc[((self.pima["glucose"] > 140) & (
            self.pima["glucose"] <= 200)), "colGlucose"] = colGlucose[1]
        self.pima.loc[self.pima["glucose"] > 200, "colGlucose"] = colGlucose[2]


    def addInsulin(self, row):
        if row["insulin"] >= 16 and row["insulin"] <= 166:
            return "Normal"
        else:
            return "Abnormal"

    
    def feature_select(self):
        self.addGlucose()
        self.addBP()
        self.pima = self.pima.assign(
            colInsulin=self.pima.apply(self.addInsulin, axis=1))
        self.pima = pd.get_dummies(
            self.pima, columns=["colGlucose", "colInsulin", "colBP"])


    def feature_def(self,x):
        model = LogisticRegression(solver='lbfgs', max_iter=10000,random_state=12345)

        self.pima = self.pima[['colGlucose_normal', 'colGlucose_prediabetes','colInsulin_Normal','colInsulin_Abnormal', 'colBP_low','colBP_high',
                                 'pregnant', 'pedigree', 'insulin', 'skin', 'bp', 'age', 'label']]
        array = self.pima.values
        
        X = array[:, 0:12]
        y = array[:, 12]
        # print(x)
        rfe = RFE(model, n_features_to_select=x)
        fit = rfe.fit(X, y)
        reduced_dataset = self.pima.iloc[:, :-1].loc[:, fit.support_]
        return reduced_dataset, self.pima.label
    

if __name__ == "__main__":
    classifer = DiabetesClassifier()
    # print(classifer.pima)
    classifer.feature_select()
    print('| Experiment | Accuracy | Confusion matrix | Comment |')
    print('|------------|----------|------------------|---------|')
    for x in [6,7,8] :
        classifer.feature_def(x)
        result = classifer.predict(x)
        # print(f"Predicition={result}")
        score = classifer.calculate_accuracy(result)
        # print(f"score={score}")
        con_matrix = classifer.confusion_matrix(result)
        # print(f"confusion_matrix=${con_matrix}")
        print(f"|Solution{x-5}   |{score}|{con_matrix}|")