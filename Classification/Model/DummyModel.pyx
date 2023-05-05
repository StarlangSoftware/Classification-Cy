from Classification.Instance.CompositeInstance cimport CompositeInstance
from Classification.InstanceList.InstanceList cimport InstanceList


cdef class DummyModel(Model):

    cpdef constructor1(self, InstanceList trainSet):
        """
        Constructor which sets the distribution using the given InstanceList.

        PARAMETERS
        ----------
        trainSet : InstanceList
            InstanceList which is used to get the class distribution.
        """
        self.distribution = trainSet.classDistribution()

    cpdef constructor2(self, str fileName):
        cdef object inputFile
        cdef int size, i, count, j
        cdef str line
        cdef list items
        inputFile = open(fileName, mode='r', encoding='utf-8')
        self.distribution = Model.loadClassDistribution(inputFile)
        inputFile.close()

    def __init__(self, trainSet: object):
        if isinstance(trainSet, InstanceList):
            self.constructor1(trainSet)
        elif isinstance(trainSet, str):
            self.constructor2(trainSet)

    cpdef str predict(self, Instance instance):
        """
        The predict method takes an Instance as an input and returns the entry of distribution which has the maximum
        value.

        PARAMETERS
        ----------
        instance : Instance
            Instance to make prediction.

        RETURNS
        -------
        str
            The entry of distribution which has the maximum value.
        """
        cdef list possible_class_labels
        if isinstance(instance, CompositeInstance):
            possible_class_labels = instance.getPossibleClassLabels()
            return self.distribution.getMaxItemIncludeTheseOnly(possible_class_labels)
        else:
            return self.distribution.getMaxItem()

    cpdef dict predictProbability(self, Instance instance):
        return self.distribution.getProbabilityDistribution()
