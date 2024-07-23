import pandas as pd
import os

# Load data
train = pd.read_csv("./data/raw/train.csv")
test = pd.read_csv("./data/raw/test.csv")

# Map 'CourseCategory' to numerical values
train['CourseCategory'] = train['CourseCategory'].map({"Business": 0, "Health": 1, "Science": 2, "Programming": 3, "Arts": 4})
test['CourseCategory'] = test['CourseCategory'].map({"Business": 0, "Health": 1, "Science": 2, "Programming": 3, "Arts": 4})

# Check if 'CourseCompletion' exists
if 'CourseCompletion' in train.columns:
    x_train = train.drop(["UserID", "CourseCompletion"], axis=1)  # Drop columns
    y_train = train["CourseCompletion"]  # This is a Series
else:
    raise KeyError("'CourseCompletion' column not found in training data.")

if 'CourseCompletion' in test.columns:
    x_test = test.drop(["UserID", "CourseCompletion"], axis=1)  # Drop columns
    y_test = test["CourseCompletion"]  # This is a Series
else:
    raise KeyError("'CourseCompletion' column not found in test data.")

# Convert y_train and y_test to DataFrames
y_train_df = y_train.to_frame()
y_test_df = y_test.to_frame()

# Combine features and target into a single DataFrame
train_df = pd.concat([x_train, y_train_df], axis=1)
test_df = pd.concat([x_test, y_test_df], axis=1)

# Display the first few rows of the train and test DataFrames
print(train_df.head())
print(test_df.head())








data_path = os.path.join("data","processed")

os.makedirs(data_path)

train_df.to_csv(os.path.join(data_path,"train_processed.csv"))
test_df.to_csv(os.path.join(data_path,"test_processed.csv"))




