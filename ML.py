import pandas as pd
import io
from google.colab import files

uploaded = files.upload()

import numpy as np

cols_x = [0,1,2,3,4,5,6,7,8,9,10]
cols_y = [11]
X = np.genfromtxt("winequality-white.csv", delimiter= ";", usecols = cols_x, skip_header=1)
Y = np.genfromtxt("winequality-white.csv", delimiter= ";", usecols = cols_y, skip_header=1)
for i in range(0, len(Y)):
  if (Y[i] <= 5):
    Y[i] = 0
   else:
    Y[i] = 1
print(X.shape, Y.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state=0)
model = LogisticRegression(max_iter = 100000)
model.fit(x_train, y_train)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
intercept_scaling=1, l1_ratio=None, max_iter=100000, multi_class='auto', n_jobs=None, penalty='l2', random_state=None, solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)

                   from sklearn.metrics import confusion_matrix, classification_report
                   model_pred = model.predict(x_test)

                   matrix = confusion_matrix(y_test, model_pred)
                   print(matrix)
                   print(model.score(x_test,y_test))
                   from sklearn.decomposition import PCA
                   pca = PCA(n_components = 5)
                   pca.fit(X)
                   print(pca.explained_variance_ratio_)
                   X2 = pca.transform(X)

                   x_train2, x_test2, y_train2, y_test2 = train_test_split(X2, Y, test_size = 0.25, random_state = 0)
                   model2 = LogisticRegression(max_iter= 1000)
                   model2.fit(x_train2, y_train2)

                   model_pred2 = model2.predict(x_test2)

                   matrix2 = confusion_matrix(y_test2, model_pred2)
                   print(matrix2)
                   print(model2.score(x_test2, y_test2))
