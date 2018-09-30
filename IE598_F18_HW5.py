import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity', 'Magnesium', "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", 
                   "Hue", "OD280/OD315 of diluted wines","Proline"]
iris = load_iris()

'''
#EDA Phase
names = df_wine.drop('Class', axis=1).columns
sns.pairplot(df_wine.iloc[:, 1:], size=2.5, vars=names)
plt.tight_layout()
plt.show()

#Wine Heatmap
cm = np.corrcoef(df_wine.iloc[:, 1:].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=False, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=names, xticklabels=names)
plt.show()

#Iris Heatmap
cm = np.corrcoef(iris.data.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=iris.feature_names, xticklabels=iris.feature_names)
plt.show()
'''

#Iris Dataset Loading
#X = iris.data
#y = iris.target
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


#80/20 Wine Dataset Train Test Split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


#Standardize the Features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


#Part 2 -- Logistic Regression and SVM
#Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train_std, y_train)

logreg_test_pred = logreg.predict(X_test_std)
logreg_train_pred = logreg.predict(X_train_std)


#print(accuracy_score(y_train, logreg_train_pred)) #same as below 
print("Logistic Regression Train Score: %.5f " % logreg.score(X_train_std, y_train))
print("Logistic Regression Test Score: %.5f" % logreg.score(X_test_std, y_test))

#Support Vector Mahcine SVM
svm = SVC()
svm.fit(X_train_std, y_train)
svm_test_pred = svm.predict(X_test_std)
svm_train_pred = svm.predict(X_train_std)

print("SVM Classifier Train Score: %.5f " % svm.score(X_train_std, y_train))
print("SVM Classifier Test Score: %.5f" % svm.score(X_test_std, y_test))


#Part 3 -- PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

'''
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
xs = X_train_pca[:,0]
ys = X_train_pca[:,1]
plt.scatter(xs, ys)
plt.axis('equal')
'''

logreg.fit(X_train_pca, y_train)
logreg_test_pred = logreg.predict(X_test_pca)
logreg_train_pred = logreg.predict(X_train_pca)

print("(PCA) Logistic Regression Train Score: %.5f " % logreg.score(X_train_pca, y_train))
print("(PCA) Logistic Regression Test Score: %.5f" % logreg.score(X_test_pca, y_test))


svm.fit(X_train_pca, y_train)
svm_test_pred = svm.predict(X_test_pca)
svm_train_pred = svm.predict(X_train_pca)

print("(PCA) SVM Classifier Train Score: %.5f " % svm.score(X_train_pca, y_train))
print("(PCA) SVM Classifier Test Score: %.5f" % svm.score(X_test_pca, y_test))


#Part 4 -- LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

logreg.fit(X_train_lda, y_train)
logreg_test_pred = logreg.predict(X_test_lda)
logreg_train_pred = logreg.predict(X_train_lda)


print("(LDA) Logistic Regression Train Score: %.5f " % logreg.score(X_train_lda, y_train))
print("(LDA) Logistic Regression Test Score: %.5f" % logreg.score(X_test_lda, y_test))


svm.fit(X_train_lda, y_train)
svm_test_pred = svm.predict(X_test_lda)
svm_train_pred = svm.predict(X_train_lda)

print("(LDA) SVM Classifier Train Score: %.5f " % svm.score(X_train_lda, y_train))
print("(LDA) SVM Classifier Test Score: %.5f" % svm.score(X_test_lda, y_test))


#Part 5 -- kPCA
gammas = [0.01, .1, 1, 10, 100]
for gam in gammas:
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=gam) #Test different gamma values
    X_train_kpca = kpca.fit_transform(X_train_std)
    X_test_kpca = kpca.transform(X_test_std)
    
    logreg.fit(X_train_kpca, y_train)
    logreg_test_pred = logreg.predict(X_test_kpca)
    logreg_train_pred = logreg.predict(X_train_kpca)
    
    print("Using gamma of ", gam)
    
    print("(kPCA) Logistic Regression Train Score: %.5f " % logreg.score(X_train_kpca, y_train))
    print("(kPCA) Logistic Regression Test Score: %.5f" % logreg.score(X_test_kpca, y_test))
    
    
    svm.fit(X_train_kpca, y_train)
    svm_test_pred = svm.predict(X_test_kpca)
    svm_train_pred = svm.predict(X_train_kpca)
    
    print("(kPCA) SVM Classifier Train Score: %.5f " % svm.score(X_train_kpca, y_train))
    print("(kPCA) SVM Classifier Test Score: %.5f" % svm.score(X_test_kpca, y_test))
    
print("My name is Stephen Pretto")
print("My NetID is: spretto2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")