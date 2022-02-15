
from numpy import *
import pandas as pd
import operator
import csv
#读取csv文件
def loadTrainData():
    l=[]
    with open('train.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line) #42001*785
    l.remove(l[0])
    l=array(l)
    label=l[:,0]
    data=l[:,1:]
    return nomalizing(toInt(data)), toInt(label)

#toTnt()函数 将字符串转换为整数，因为从csv文件读取出来的，是字符串类型的，比如‘253’，而我们接下来运算需要的是整数类型的，因此要转换，int(‘253’)=253。
def toInt(array):
    array=mat(array)  #调用mat()函数可以将数组转换为矩阵，然后可以对矩阵进行一些线性代数的操作。
    m,n=shape(array)
    newArray=zeros((m,n))
    for i in range(m):
        for j in range(n):
            newArray[i,j]=int(array[i,j])
    return newArray

#nomalizing()函数做的工作是归一化，因为train.csv里面提供的表示图像的数据是0～255的，为了简化运算，我们可以将其转化为二值图像，因此将所有非0的数字，即1～255都归一化为1。nomalizing()函数如下：
def nomalizing(array):
    m,n=shape(array)
    for i in range(m):
        for j in range(n):
            if array[i,j]!=0:
                array[i, j] = 1
    return array

#test.csv里的数据大小是28001*784，第一行是文字描述，因此实际的测试数据样本是28000*784，与train.csv不同，没有label，28000*784即28000个测试样本，我们要做的工作就是为这28000个测试样本找出正确的label。所以从test.csv我们可以得到测试样本集testData，代码如下：
def loadTestData():
    l=[]
    with open('test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    data=array(l)
    return nomalizing(toInt(data))

#前面已经提到，由于digit recognition是训练赛，所以这个文件是官方给出的参考结果，本来可以不理这个文件的，但是我下面为了对比自己的训练结果，所以也把sample_.csv这个文件读取出来，这个文件里的数据是28001*2，第一行是文字说明，可以去掉，第一列表示图片序号1～28000，第二列是图片对应的数字。从knn_benchmark.csv可以得到28000*1的测试结果矩阵testResult，代码：

def loadTestResult():
    l=[]
    with open('sample_submission.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line)
     #28001*2
    l.remove(l[0])
    label=array(l)
    return toInt(label[:,1])

#到这里，数据分析和处理已经完成，我们获得的矩阵有：trainData、trainLabel、testData、testResult


#接下来采用KNN算法来分类
#简单说明一下，inX就是输入的单个样本，是一个特征向量。dataSet是训练样本，对应上面的trainData，labels对应trainLabel，k是knn算法选定的k，一般选择0～20之间的数字。这个函数将返回inX的label，即图片inX对应的数字。
#对于测试集里28000个样本，调用28000次这个函数即可。

def classify(inX, dataSet, labels, k):
    inX=mat(inX)
    dataSet=mat(dataSet)
    labels=mat(labels)
    dataSetSize = dataSet.shape[0]  #shape[0]得出dataSet的行数即样本个数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet #tile(A,(m,n))将数组A作为元素构造m行n列的数组
    sqDiffMat = array(diffMat)**2
    sqDistances = sqDiffMat.sum(axis=1)  #array.sum(axis=1)按行累加，axis=0为按列累加
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()#array.argsort()，得到每个元素的排序序号
    classCount={} #sortedDistIndicies[0]表示排序后排在第一个的那个数在原来数组中的下标
    for i in range(k):
        voteIlabel = labels[0,sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#上面我们得到了28000个测试样本的label，必须将其保存成csv格式文件才可以提交
def saveResult(result):
    with open('result.csv','wb') as myFile:
        myWriter=csv.writer(myFile)
        for i in result:
            tmp=[]
            tmp.append(i)
            myWriter.writerow(tmp)

#上面各个函数已经做完了所有需要做的工作，现在需要写一个函数将它们组合起来解决digit recognition这个题目。我们写一个handwritingClassTest函数，运行这个函数，就可以得到训练结果result.csv。
def handwritingClassTest():
    trainData,trainLabel=loadTrainData()
    testData=loadTestData()
    testLabel=loadTestResult()
    m,n=shape(testData)
    errorCount=0
    resultList=[]
    for i in range(m):
         classifierResult = classify(testData[i], trainData, trainLabel, 5)
         resultList.append(classifierResult)
         print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, testLabel[0,i]))
         if (classifierResult != testLabel[0,i]): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(m)))
    saveResult(resultList)


handwritingClassTest()