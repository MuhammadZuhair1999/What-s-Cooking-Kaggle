# Imports
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from collections import Counter
import matplotlib as mpl
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
 
# Read the data
df = pd.read_json('train.json')
df.head()

# Create empty list to store recipe features
features_all_list = []

# Extract the features from each recipe (need a global list)
for i in df.ingredients:
    features_all_list += i
    
# Remove duplicate features using default set behavior
features = list( set(features_all_list) )

len(features)

# Create a zeros-only matrix with a row for each recipe and column for each feature
onehot_ingredients = np.zeros((df.shape[0], len(features)))

# Index the features (ingredients) alphabetically
feature_lookup = sorted(features)

# For each recipe look up ingredient position in the sorted ingredient list
# If that ingredient exists, set the appropriate column equal to 1
## This will take 1-2 minutes to finish running
for index, row in df.iterrows():
    for ingredient in row['ingredients']:
        onehot_ingredients[index, feature_lookup.index(ingredient)] = 1

y = df.cuisine.values.reshape(-1,1)

# Create a dataframe
df_features = pd.DataFrame(onehot_ingredients)

# Create empty dictionary to store featureindex:columnname
d = {}

# For each feature, fetch the column name
for i in range(len(features)):
    d[df_features.columns[i]] = features[i]

# Rename the features (stop using the index # and use the actual text)
df_features = df_features.rename(columns=d)
df_features.shape

# Import train_test_split
from sklearn.model_selection import train_test_split

# Split into train, test
X_train, X_test, y_train, y_test = train_test_split(df_features, y, test_size=0.2, shuffle=True, random_state=42)

# Import decision tree from sklearn
from sklearn.tree import DecisionTreeClassifier

# Set up the decision tree
clf = DecisionTreeClassifier(max_features=5000)
 
# Fit the decision tree to the training data
clf.fit(X_train, y_train)

# Use the decision tree to predict values for the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy score and print the results
a = accuracy_score(y_test, y_pred)
print("Accuracy Score in % : ")
print(a * 100)

# Set up random forest classifier
clf = RandomForestClassifier()

# Train the random forest (use ravel to coerce to 1d array)
clf.fit(X_train, y_train.ravel())

# Get test predictions
y_pred = clf.predict(X_test)

# Get accuracy for the random forest classifier
a = accuracy_score(y_test, y_pred)
print("Accuracy Score in % : ")
print(a * 100)

# Setting up the tuned random forest
clf = RandomForestClassifier(max_depth=200, n_estimators=250, max_features='sqrt', min_samples_split=7)

# Set up and fitlogistic regression
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train.ravel())

# Get predictions on test data
y_pred = clf.predict(X_test)

# Get accuracy
a = accuracy_score(y_test, y_pred)
print("Accuracy Score in % : ")
print(a * 100)

# Group by cuisine and aggregate the data
data_agg = df.groupby('cuisine').apply(lambda x: x.sum())
data_agg = data_agg.drop(columns=['cuisine','id'])
data_agg = data_agg.reset_index()
 
## Get all of the unique ingredients as features
features_all_list = []

for i in df.ingredients:
    features_all_list += i
    
features = list(set(features_all_list))
len(features)

onehot_ingredients = np.zeros((data_agg.shape[0], len(features)))
feature_lookup = sorted(features)

# Set # of clusters
## We tried 3, 4, 5, 6, 7, 8, and 10 with 5 being the best
numOfClusters = 5

# Set up KMeans
kmeans = KMeans(init='k-means++', n_clusters=numOfClusters, n_init=10)

# Fit kmeans
kmeans.fit(reduced_data)

# Predict kmeans
kmeans_pred = kmeans.predict(reduced_data)

# Generate plot of the resultant clusters
x = reduced_data[:, 0]
y = reduced_data[:, 1]

# Set font size
plt.rcParams.update({'font.size':15
                    })

# Get fig, ax, and set figure size
fig, ax = plt.subplots(figsize=(10,10))

# Scatter the cuisines
ax.scatter(x, y, s=5000, c=kmeans_pred, cmap='Set3')

# Add labels to each cuisine
for i, txt in enumerate(data_agg.cuisine):
    ax.annotate(txt, (x[i], y[i]))
    
    