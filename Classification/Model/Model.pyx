from io import TextIOWrapper

from DataStructure.CounterHashMap cimport CounterHashMap
from Math.DiscreteDistribution cimport DiscreteDistribution
from Math.Matrix cimport Matrix

cdef class Model(object):

    cpdef str predict(self, Instance instance):
        """
         An abstract predict method that takes an Instance as an input.

        PARAMETERS
        ----------
        instance : Instance
            Instance to make prediction.

        RETURNS
        -------
        str
            The class label as a String.
        """
        pass

    cpdef dict predictProbability(self, Instance instance):
        pass

    cpdef Instance loadInstance(self, str line, list attributeTypes):
        cdef list items
        cdef Instance instance
        cdef int i
        items = line.split(",")
        instance = Instance(items[len(items) - 1])
        for i in range(len(items) - 1):
            if attributeTypes[i] == "DISCRETE":
                instance.addDiscreteAttribute(items[i])
            elif attributeTypes[i] == "CONTINUOUS":
                instance.addContinuousAttribute(float(items[i]))
        return instance

    cpdef Matrix loadMatrix(self, object inputFile):
        cdef Matrix matrix
        cdef int j, k
        cdef str line
        cdef list items
        items = inputFile.readline().strip().split(" ")
        matrix = Matrix(int(items[0]), int(items[1]))
        for j in range(matrix.getRow()):
            line = inputFile.readline().strip()
            items = line.split(" ")
            for k in range(matrix.getColumn()):
                matrix.setValue(j, k, float(items[k]))
        return matrix

    @staticmethod
    def loadClassDistribution(inputFile: TextIOWrapper) -> DiscreteDistribution:
        distribution = DiscreteDistribution()
        size = int(inputFile.readline().strip())
        for i in range(size):
            line = inputFile.readline().strip()
            items = line.split(" ")
            count = int(items[1])
            for j in range(count):
                distribution.addItem(items[0])
        return distribution

    @staticmethod
    def getMaximum(classLabels: list) -> str:
        """
        Given an array of class labels, returns the maximum occurred one.

        PARAMETERS
        ----------
        classLabels : list
            An array of class labels.

        RETURNS
        -------
        str
            The class label that occurs most in the array of class labels (mod of class label list).
        """
        frequencies = CounterHashMap()
        for label in classLabels:
            frequencies.put(label)
        return frequencies.max()
