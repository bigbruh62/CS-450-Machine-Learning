import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")


columns = "Age workclass fnlwgt education_level education_num" \
          " marital_status occupation relationship_status race" \
          " sex capital_gain capital_loss hours_per_week " \
          "native_country salary".split()
data.columns = columns

data = data.replace(to_replace=" ?", value="NaN")
data.dropna(inplace=True)
#print(data)

#transform to all numerical data
cols_to_transform = ["workclass", "education_level", "marital_status", "occupation", "relationship_status", "race", "sex", "native_country", "salary"]
df_with_dummies = pd.get_dummies(data=data, columns=cols_to_transform)

print(df_with_dummies)

data_as_np_array = df_with_dummies.as_matrix()
x_train, x_test, y_train, y_test = train_test_split(data_as_np_array, )
# zscore of data
#data_scaled = preprocessing.scale(data)
#print(data_scaled)
#numeric_cols = data.select_dtypes(include=[np.number]).columns
#zscore_data = data[numeric_cols].apply(zscore)
#print(zscore_data)

