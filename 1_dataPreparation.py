import pandas as pd

# Correctly formatted file path
df = pd.read_csv(r"C:\4-Data_Preparation\ethnic diversity.csv")

# Check data types
print(df.dtypes)

# Convert 'salaries' column to integer
df['salaries'] = df['salaries'].astype(int)

# Check data types again
print(df.dtypes)

# Convert 'age' column to float
df['age'] = df['age'].astype(float)

# Check data types one more time
print(df.dtypes)

#-------------------------------------------------------------------------------------------------------
import pandas as pd

# Read the CSV file into a DataFrame
df_new = pd.read_csv("education.csv")

# Check for duplicates in the DataFrame
duplicate = df_new.duplicated()

# Sum the number of duplicate rows
num_duplicates = duplicate.sum()

# Print the number of duplicates
print(f"Number of duplicate rows: {num_duplicates}")


#-------------------------------------------------------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv(r"C:\4-Data_Preparation\ethnic diversity.csv")
print(df.columns)

# Visualize the 'Salaries' column using a boxplot
plt.figure(figsize=(10, 5))

# Boxplot for 'Salaries'
plt.subplot(1, 2, 1)
sns.boxplot(df['Salaries'])
plt.title('Boxplot of Salaries')

# Boxplot for 'age'
plt.subplot(1, 2, 2)
sns.boxplot(df['age'])
plt.title('Boxplot of Age')

plt.tight_layout()
plt.show()

# Calculate the IQR for 'Salaries'
Q1 = df['Salaries'].quantile(0.25)
Q3 = df['Salaries'].quantile(0.75)
IQR = Q3 - Q1

# Print the IQR value
print(f"Interquartile Range (IQR) for salaries: {IQR}")

# Calculate the lower and upper limits
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Print the lower and upper limits
print(f"Lower limit for salaries: {lower_limit}")
print(f"Upper limit for salaries: {upper_limit}")

#------------------------------------------------------------------------------------------------------------

#trimming
import numpy as np
outliers_df=np.where(df.Salaries>upper_limit,True,np.where(df.Salaries<lower_limit,True,False))
df_trimmed=df.loc[~outliers_df]
df.shape
#(310, 13)
df_trimmed.shape
#(306, 13)
sns.boxplot(df_trimmed.Salaries)

#drawback :in trimming we may loose our data
#------------------------------------------------------------------------------------------------------------

#replacement technique
df = pd.read_csv(r"C:\4-Data_Preparation\ethnic diversity.csv")
df.describe()

#record no. 23 has got outliers
#if value >upper limit map it to upper limit,simlarly if it lower limit ma it to lower limit 
df_replaced=pd.DataFrame(np.where(df.Salaries>upper_limit,upper_limit,np.where(df.Salaries<lower_limit,lower_limit,df.Salaries)))
sns.boxplot(df_replaced[0])

#------------------------------------------------------------
#Winsorization is a statistical transformation technique used to limit extreme values in the data to reduce the effect of possibly spurious outliers.
#pip install feature_engine
from feature_engine.outliers import Winsorizer
# Initialize the Winsorizer with the correct column name
winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['Salaries'])


df_t=winsor.fit_transform(df[['Salaries']])
sns.boxplot(df['Salaries'])
sns.boxplot(df_t['Salaries'])

'''
#example of winsorization
Original Data:
    Salaries
0     50000
1     52000
2     48000
3     60000
4     70000
5     80000
6     90000
7     95000
8   1000000

Winsorized Data:
    Salaries  Salaries_winsorized
0     50000                50000
1     52000                52000
2     48000                48000
3     60000                60000
4     70000                70000
5     80000                80000
6     90000                90000
7     95000                95000
8   1000000                95000

In the Winsorized data, the extreme value (1000000) 
is replaced with the value at the 95th percentile 
(95000), demonstrating how Winsorization can 
effectively reduce the impact of outliers.

'''














