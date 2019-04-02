import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

def predict():
    rng=np.random.RandomState(1)        #随机数种子
    X=np.sort(5*rng.rand(80,1),axis=0)
    y=np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(16))    #y[::5]说明遍历所有数据，以步进为5

    regr_1=DecisionTreeRegressor(max_depth=2)
    regr_2=DecisionTreeRegressor(max_depth=5)
    regr_1.fit(X,y)
    regr_2.fit(X,y)

    #predict
    X_test=np.arange(0.0,5.0,0.01)[:,np.newaxis]
    y_1=regr_1.predict(X_test)
    y_2=regr_2.predict(X_test)
    return X,y,X_test,y_1,y_2

def plotFigure(X,y,X_test,y_1,y_2):
    plt.figure()
    plt.scatter(X,y,s=20,edgecolors='black',c='darkorange',label='data')
    plt.plot(X_test, y_1, color="cornflowerblue",
             label="max_depth=2", linewidth=2)
    plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()

if __name__=='__main__':
    X,y,X_test,y_1,y_2=predict()
    plotFigure(X,y,X_test,y_1,y_2)
