from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
import os

k = 2

def autoNorm(dataSet):
	#return the minimum along a given axis
	#here 'along rows'
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals-minVals
	normDataSet = zeros(shape(dataSet))
	#number of rows
	m = dataSet.shape[0]
	#minVals of shape [1, 3], use tile to expand it to [m, 3]
	normDataSet = dataSet - tile(minVals, (m, 1))
	#here / means corresponding division
	normDataSet = normDataSet/tile(ranges, (m, 1))
	return normDataSet, ranges, minVals

def file2matrix(filename):
	fr = open(filename, 'r')
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = zeros((numberOfLines,3))
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		line = line.strip()
		listFromLine = line.split('\t')
		#string here, but numpy.array will take care of it
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat, classLabelVector

def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	#axis = 1, sum along columns
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	#return corresponding index in sorted order of value
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in xrange(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) +1
	sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]

def datingClassTest():
	global k
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], k)
		print "the classifier came back with: %d, the real answer is: %d"%(classifierResult, datingLabels[i])
		if (classifierResult!=datingLabels[i]):
			errorCount += 1.0
	print "the total error rate is: %f"%(errorCount/float(numTestVecs),)

def classifyPerson():
	global k
	resultList = ['not at all', 'in small doses', 'in large doses']
	percentTats = float(raw_input("percentage of time spent playing video game?"))
	ffMiles = float(raw_input("frequent flier miles earned per year?"))
	iceCream = float(raw_input("liters of ice cream consumed per year?"))
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([percentTats, ffMiles, iceCream])
	classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, k)
	print "You will probably like this person: ", resultList[classifierResult-1]

def img2vector(filename):
	returnVect = zeros((1, 1024))
	fr = open(filename)
	for i in xrange(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	return returnVect

def handwritingClassTest():
	global k
	hwLabels = []
	trainingFileList = os.listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m, 1024))
	for i in xrange(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s'%(fileNameStr,))
	testFileList = os.listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in xrange(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s'%(fileNameStr,))
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, k)
		print "the classifier came back with: %d, the real answer is: %d"%(classifierResult, classNumStr)
		if classifierResult != classNumStr:
			errorCount += 1
	print "\nthe total number of errors is: %d"% errorCount
	print "\nthe total error rate is: %f"%(errorCount/float(mTest),)

if __name__ == "__main__":
	group, labels = createDataSet()
	datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
	fig = plt.figure()	
	ax = fig.add_subplot(111)
	#color and size change as labels change
	ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0*array(datingLabels), 15.0*array(datingLabels))
	normMat, ranges, minVals = autoNorm(datingDataMat)
	#plt.show()
	#datingClassTest()
	#classifyPerson()
	handwritingClassTest()
