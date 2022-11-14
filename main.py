import pandas as pd
import numpy as np

#Making dataframe from dataset
#Columns --> age,sex,bmi,children,smoker,region,charges

# creating a data frame
insuranceData = pd.read_csv("insurance.csv")
print(insuranceData.head())