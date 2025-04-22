## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![Screenshot 2025-04-22 151309](https://github.com/user-attachments/assets/27675c9a-f2c8-4561-bd55-b36f2ad8229a)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2025-04-22 151315](https://github.com/user-attachments/assets/c5dcdc32-85fd-47c9-aef8-067fe196fc09)

```
 df['bo2']=e1.fit_transform(df[["ord_2"]])
 df
```
![Screenshot 2025-04-22 151322](https://github.com/user-attachments/assets/f756586b-4efa-47ac-866d-3c460d43bb4f)

```
 le=LabelEncoder()
 dfc=df.copy()
 dfc['ord_2']=le.fit_transform(dfc['ord_2'])
 dfc
```
![Screenshot 2025-04-22 151328](https://github.com/user-attachments/assets/bee61f6b-151c-4582-bc35-adf03fb4d802)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![Screenshot 2025-04-22 151335](https://github.com/user-attachments/assets/bf34a460-361c-4172-9dc3-04d7ce8d36f7)

```
 pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2025-04-22 151343](https://github.com/user-attachments/assets/6eeeebd5-14de-4497-b3e9-d20778d15e34)

```
 pip install --upgrade category_encoders
```
![Screenshot 2025-04-22 151356](https://github.com/user-attachments/assets/059fa57f-292c-4f9e-aedf-b3b7008b9970)

```
 from category_encoders import BinaryEncoder
 df=pd.read_csv("data.csv")
 df
```
![Screenshot 2025-04-22 151403](https://github.com/user-attachments/assets/5ef861a5-5bd3-4d07-a55a-8102f217682d)

```
 be=BinaryEncoder()
 nd=be.fit_transform(df['Ord_2'])
 dfb=pd.concat([df,nd],axis=1)
 dfb
```
![Screenshot 2025-04-22 151410](https://github.com/user-attachments/assets/307d8f5f-3156-41aa-b3f3-bf86c563bd25)

```
 from category_encoders import TargetEncoder
 te=TargetEncoder()
 CC=df.copy()
 new=te.fit_transform(X=CC["City"],y=CC["Target"])
 CC=pd.concat([CC,new],axis=1)
 CC
```
![Screenshot 2025-04-22 151417](https://github.com/user-attachments/assets/51df73d0-8662-429b-8f88-0f815ac6191b)

```
 import pandas as pd
 from scipy import stats
 import numpy as np
 df=pd.read_csv("Data_to_Transform.csv")
 df
```
![Screenshot 2025-04-22 151425](https://github.com/user-attachments/assets/4d623555-5192-4809-b1c2-13d8729072ee)

```
df.skew()
```
![Screenshot 2025-04-22 151433](https://github.com/user-attachments/assets/70515196-235d-4e45-b504-9da99ab3208e)

```
np.log(df["Highly Positive Skew"])
```
![Screenshot 2025-04-22 151439](https://github.com/user-attachments/assets/bf96d72b-e10f-4838-bb8f-9d1c4514e968)

```
 np.reciprocal(df["Moderate Positive Skew"])
```
![Screenshot 2025-04-22 151446](https://github.com/user-attachments/assets/27a61556-6972-47bc-aec4-74a45062ae10)

```
np.sqrt(df["Highly Positive Skew"])
```
![Screenshot 2025-04-22 151452](https://github.com/user-attachments/assets/8eddbac8-c189-4b36-98ae-d534d843617f)

```
 np.square(df["Highly Positive Skew"])
```
![Screenshot 2025-04-22 151459](https://github.com/user-attachments/assets/e4fb7a19-5e95-4e3d-818e-ce21227416ff)

```
 df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
 df
```
![Screenshot 2025-04-22 151508](https://github.com/user-attachments/assets/7895c32f-cece-4240-a73a-5ba0210e13f4)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![Screenshot 2025-04-22 151514](https://github.com/user-attachments/assets/af5b7e21-deb9-4161-a3a4-afb6f8cc8e24)

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 df
```
![Screenshot 2025-04-22 151523](https://github.com/user-attachments/assets/ccf7b51d-6ac3-4dbb-a7ee-153009bdfa50)

```
 import seaborn as sns
 import statsmodels.api as sm
 import matplotlib.pyplot as plt
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```
![Screenshot 2025-04-22 151531](https://github.com/user-attachments/assets/bf0132dc-0c43-42e9-a303-4390c7ad4719)

```
 sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
 plt.show()
```
![Screenshot 2025-04-22 151538](https://github.com/user-attachments/assets/3239a717-54f4-4b6e-a87e-1798ba772480)

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```
![Screenshot 2025-04-22 151546](https://github.com/user-attachments/assets/a6a1d116-79d2-4148-b1b3-db6a55c01772)


# RESULT:
       # INCLUDE YOUR RESULT HERE

       
