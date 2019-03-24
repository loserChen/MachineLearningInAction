import numpy as np
import operator
'''
利用简单数据集进行分类
'''
def createDataSet():
    '''
    创建数据集
    :return:无
    '''
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels
def classify0(inX,dataSet,labels,k):
    '''
    分类
    :param inX:用于分类的输入向量
    :param dataSet:输入的训练样本集
    :param labels:标签向量
    :param k:选择最近邻居的数目
    :return:
    '''
    dataSetSize=dataSet.shape[0] #获得数据集的数量
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet  #np.tile()函数就是将inX在行方向上复制dataSetSize倍，在列方向上复制1倍
    sqDiffMat=diffMat**2  #对diffMat求平方
    sqDistances=sqDiffMat.sum(axis=1)  #对列方向上的数据进行求和
    distances=sqDistances**0.5 #开根号
    sortedDistIndices=distances.argsort()  #排序，得到距离从小到大排序的序号
    classCount={} #计算类的数量的字典
    for i in range(k):
        voteIlabel=labels[sortedDistIndices[i]]  #获得欧式距离最小的k个标签
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1  #每得到一个对应的标签就在classCount上加一
    # python3中用items()替换python2中的iteritems()
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]  #返回得票数最多的label
if __name__=='__main__':
    x,y=createDataSet()
    print(classify0([1.0,1.0],x,y,3))
