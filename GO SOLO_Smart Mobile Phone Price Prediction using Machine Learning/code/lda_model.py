import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv").drop('id', axis=1)

print("Number of pricing options:", len(train['price_range'].unique()))
prices = ['Cheap', 'Budget', 'Expensive', 'Premium']
print("Pricing options:", prices)

#splitting the data
X = train.drop('price_range', axis = 1)
y = train['price_range'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=56)

#FeatureScaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Using Linear Discriminant Analysis

LDA = LinearDiscriminantAnalysis(n_components=1)

X_train_lda = LDA.fit_transform(X_train, y_train.ravel())
X_test_lda = LDA.transform(X_test)

lda_model = LogisticRegression(multi_class = 'multinomial', solver = 'sag',  max_iter = 10000)
lda_model.fit(X_train_lda, y_train.ravel())

y_pred_lda = lda_model.predict(X_test_lda)

lda_acc = accuracy_score(y_test, y_pred_lda)*100
print("\nLDA accuracy:", lda_acc)

#Testing with LDA

n = [5, 759, 42, 653]

for i in n:
	print("\nTest instance", i)
	print(test.iloc[i])

	X = sc.transform(test)
	X = LDA.transform(X)

	print("\nPredicted range:", prices[lda_model.predict(X)[i]])
