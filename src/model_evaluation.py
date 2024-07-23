import numpy as np
import pandas as pd

import pickle
import json
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score


test_df = pd.read_csv("./data/processed/test_processed.csv")



x_test = test_df.iloc[:,0:-1]
y_test = test_df.iloc[:,-1]


scaler=StandardScaler()
x_test=scaler.fit_transform(x_test)


# y_predict4=rf_model.predict(x_test)
# accuracy4=accuracy_score(y_test,y_predict4)
# print("the accuracy = ",(accuracy4*100),"%")





clf = pickle.load(open('model.pkl','rb'))
y_pred = clf.predict(x_test)



accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    
}

with open('metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent=4)