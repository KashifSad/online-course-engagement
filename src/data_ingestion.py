import pandas as pd
import os
from sklearn.model_selection import train_test_split



df = pd.read_csv(r"C:\Users\MV\Desktop\project\archive\online_course_engagement_data.csv")
print(df.head())

# x = df.iloc[:,0:-1]
# y = df.iloc[:,-1]


train_data,test_data = train_test_split(df,test_size=0.2,random_state=42)


data_path = os.path.join("data","raw")

os.makedirs(data_path)

train_data.to_csv(os.path.join(data_path,"train.csv"))
test_data.to_csv(os.path.join(data_path,"test.csv"))
