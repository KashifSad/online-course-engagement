import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle


train_df = pd.read_csv("./data/processed/train_processed.csv")


x = train_df.iloc[:,0:-1]
y = train_df.iloc[:,-1]


scaler=StandardScaler()
x=scaler.fit_transform(x)

rf_model=RandomForestClassifier()
rf_model.fit(x,y)


pickle.dump(rf_model,open("model.pkl","wb"))


