import numpy as np

def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=list(map(float,curLine))              #映射成浮点数
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet,feature,value):
    mat0=dataSet[np.nonzero(dataSet[:,feature]>value)[0],:]     #找到某个feature大于value的数据
    mat1=dataSet[np.nonzero(dataSet[:,feature]<=value)[0],:]    #找到某个feature小于等于value的数据
    return mat0,mat1

def linearSolve(dataSet):
    m,n=np.shape(dataSet)
    X=np.mat(np.ones((m,n)))
    Y=np.mat(np.ones((m,1)))
    X[:,1:n]=dataSet[:,0:n-1]
    Y=dataSet[:,-1]
    xTx=X.T*X
    if np.linalg.det(xTx)==0.0:
        raise NameError('This matrix is singular, cannot do inverse, try increasing the second value of ops')
    ws=xTx.I*(X.T*Y)
    return ws,X,Y

# def regLeaf(dataSet):
#     return np.mean(dataSet[:,-1])               #当函数确定不在对数据进行切分时，将调用该函数来得到叶结点的模型
#利用modelLeaf来代替regLeaf

def modelLeaf(dataSet):
    ws,X,Y=linearSolve(dataSet)
    return ws

# def regErr(dataSet):
#     return np.var(dataSet[:,-1])*np.shape(dataSet)[0]
#利用modelErr来代替regErr
def modelErr(dataSet):
    ws,X,Y=linearSolve(dataSet)
    yHat=X*ws
    return np.sum(np.power(Y-yHat,2))

def chooseBestSplit(dataSet,leafType=modelLeaf,errType=modelErr,ops=(1,4)):
    tolS=ops[0] #容许的误差下降值
    tolN=ops[1] #是切分的最少样本数
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:              #如果所有标签值相等即退出
        return None,leafType(dataSet)
    m,n=np.shape(dataSet)
    S=errType(dataSet)
    bestS=float('inf')
    bestIndex=0
    bestValue=0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
            mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)
            if np.shape(mat0)[0]<tolN or np.shape(mat1)[0]<tolN:
                continue
            newS=errType(mat0)+errType(mat1)
            if newS<bestS:
                bestIndex=featIndex
                bestValue=splitVal
                bestS=newS
    if S-bestS<tolS:
        return None,leafType(dataSet)
    return bestIndex,bestValue

def createTree(dataSet,leafType=modelLeaf,errType=modelErr,ops=(1,4)):
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
    if feat==None:
        return val
    retTree={}
    retTree['spInd']=feat
    retTree['spVal']=val
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    retTree['left']=createTree(lSet,leafType,errType,ops)
    retTree['right']=createTree(rSet,leafType,errType,ops)
    return retTree

if __name__=='__main__':
    myMat=np.mat(loadDataSet('exp2.txt'))
    print(createTree(myMat,modelLeaf,modelErr,(1,10)))
