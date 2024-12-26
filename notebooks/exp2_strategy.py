# creating multiple runs- standard scaler, minmax , random forest, maive bayes, logistic regression
import pandas as pd
import numpy as np
import mlflow
import dagshub


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv(r'E:\Rohini\miniproject\project123\data\external\loan_approval_dataset.csv')
df.head
df.columns = df.columns.str.strip()
x = df.drop(['loan_id', 'loan_status'], axis=1)
y= df['loan_status']
x= pd.get_dummies(x)

scalers = {'standard': StandardScaler(), 'minmax': MinMaxScaler()}
models = {'logistic_regression': LogisticRegression(), 'random_forest': RandomForestClassifier(), 'naive_bayes': GaussianNB()}

#set tracking uri
mlflow.set_tracking_uri("https://dagshub.com/rohinibhosale1223/miniproject.mlflow")

dagshub.init(repo_owner='rohinibhosale1223', repo_name='miniproject', mlflow=True)

mlflow.set_experiment('exp2_strategy1')

with mlflow.start_run() as parent_run:
    for algo, model in models.items():
        for scaler_name, scaler in scalers.items():
            #start the child run
            with mlflow.start_run(run_name=f'{algo}_{scaler_name}', nested=True) as child_run:

                x_scaled = scaler.fit_transform(x)
                x_train, x_test, y_train, y_test = train_test_split(x_scaled, y
                                                                    , test_size=0.2, random_state=42)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                # log parameters- log scaler and algo used
                mlflow.log_param('scaler', scaler_name)
                mlflow.log_param('algo', algo)
                # log metric
                mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
                mlflow.log_metric('f1', f1_score(y_test, y_pred, pos_label= ' Approved'))
                mlflow.log_metric('precision', precision_score(y_test, y_pred, pos_label=' Approved'))
                mlflow.log_metric('recall', recall_score(y_test, y_pred, pos_label=' Approved'))
                # log model
                mlflow.sklearn.log_model(model, 'model')
                # log the python file
                mlflow.log_artifact(__file__)


