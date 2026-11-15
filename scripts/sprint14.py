# %%
# INICIALIZATION

import math
from pathlib import Path
from src.preprocess import preprocess_data
import src.knn as knn

import numpy as np
import pandas as pd

import seaborn as sns

import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing

from sklearn.model_selection import train_test_split

print("versión numpy: ", np.__version__)
print("versión pandas: ", pd.__version__)
print("versión sklearn: ", sklearn.__version__)
print("versión seaborn: ", sns.__version__)
# %%


# %%
# LOADING AND INITIAL INSPECTION

# Load data, resolve path regardless of directory structure
DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'insurance_us.csv'
df = pd.read_csv(DATA_PATH)

brief, df = preprocess_data(df)

print("-"*70)
df['age']=df['age'].astype(int)
print("Updated dataframe info...")
print(df.info())

# Age is now integer. Duplicates were removed, in future work would be good to consult with
# subject matter expert because it could be that one person can have more than one policy, for example.
# For now, this is a reasonable assumption.
# %%

# %%
# VALUE COUNTS FOR INSURANCE BENEFITS
# The data for 'insurance_benefits' looks very lopsided. Let's do a value count.
print("\nValue count for 'insurance_benefits'...")
print(df['insurance_benefits'].value_counts())
# %%

# %%
# PAIRWISE PLOT
g = sns.pairplot(df, kind='hist')
g.fig.set_size_inches(12, 12)

# %% [Markdown]
# The pairwise data show a couple things:
# 1. There are no significant differences based on gender. Age range, salary range, family members all look alike. Gender 1 has a bit more variance for insurance benefits claimed, possibly due to chance. Nothing we want to delve into just now.
# 2. There is a pattern in the pair age vs insurance benefits. There is a plateau from 20 to 40 years old, then the benefits increase with age. This is relevant for the linear regression, because a straight line will necessarily lose info. We will need to account for it.
# 2.a You can already tell that age will be a significant predictor.
# 3. The distribution of age looks normal-ish, with a right skew. A lot of young people in this dataset.
# 4. The distribution of salary looks like a normal.
# 5. The distribution of insurance benefits does not look normal, the tail looks heavy. This is very relevant, we need to find out what distribution is this.
# %%

# %%
# CORRELATION MATRIX
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')

# Look at that! Age does almost all the work.
# A linear regression will over-predict benefits for the mass of zeroes. I fully expect it to perform worse than the dummy.

# %%

# %%
# Task 1: FIND SIMILAR CUSTOMERS TO A GIVEN QUERY.

features = ['gender', 'age', 'salary', 'family_members']
target = 'insurance_benefits'

X_full = df[features]

# From the basic stats, we know that field 'salary' is many orders of magnitde larger all other fields.
# The effect will be that the nearest-neighbors function is basically a salary match function.
# However, we will strive to turn these assertions into formal proof.

# Case 1: Extreme age, median salary, median family members
query1 = {
    'gender': 1,
    'age': 50,
    'salary': 40200,
    'family_members': 1
}

# Case 2: Median age, median salary, extreme family members
query2 = {
    'gender': 0,
    'age': 30,
    'salary': 40200,
    'family_members': 6
}

# Find similar customers, unscaled data, straight line and city block distances.
# Where 'straight line' is the euclidean distance, and 'city block' is the Manhattan distance.

for query in [query1, query2]:
    neighbors_d1 = knn.find_like(X_full, query, 'straight line')
    print(neighbors_d1)

    neighbors_d2 = knn.find_like(X_full, query, 'city block')
    print(neighbors_d2)

# The cases yield the expected output, i.e. mostly salary match.
# Now, we will take a random sample from the data, build queries around it and evaluate to what extent the salary impacts the distance calculation.

print("\n",knn.contributions(neighbors_d1, query1))
print("\n",knn.contributions(neighbors_d2, query1))

print("\n",knn.contributions(neighbors_d1, query2))
print("\n",knn.contributions(neighbors_d2, query2))


# # Take a random sample from the data.
# random_sample = df.sample(n=1, random_state=42)[features].iloc[0]
# print('\nTest effects of salary on distance calculation for unscaled data...')
# print("Random sample: ",random_sample.to_dict())

# # Build a query around the sample. Equal salary, large difference in age and family members.
# query3 = random_sample.to_dict()
# query3['age'] = query3['age'] + 20
# query3['family_members'] = query3['family_members'] + 4

# # Build another query around the sample. Large difference in salary, equal age and family members.
# query4 = random_sample.to_dict()
# query4['salary'] = query4['salary'] + 10000

# for query in [query3, query4]:
#     print("\nQuery: ", query)
#     print("Straight line distance: ", knn.get_distance(query, random_sample, 'straight line', features))
#     print("City block distance : ", knn.get_distance(query, random_sample, 'city block', features))

# This is sufficient proof that salary is the dominant factor in the distance calculation.

# %%
# Task 1(cont.): SCALED DATASET
# Now, we will scale the data and see if the results change.

# Scale the data.
scaler = sklearn.preprocessing.MaxAbsScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_full), columns=features)

# Scale the queries.
query1_scaled = dict(zip(features, scaler.transform(pd.DataFrame([query1]))[0]))
query2_scaled = dict(zip(features, scaler.transform(pd.DataFrame([query2]))[0]))

for query in [query1_scaled, query2_scaled]:
    neighbors_scaled_d1 = knn.find_like(X_scaled, query, 'straight line')
    print(X_full.loc[neighbors_scaled_d1.index])

    neighbors_scaled_d2 = knn.find_like(X_scaled, query, 'city block')
    print(X_full.loc[neighbors_scaled_d2.index])


# %%

# %%
# %%
