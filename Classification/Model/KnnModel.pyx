from functools import cmp_to_key
from Classification.Model.KnnInstance cimport KnnInstance
from Classification.Instance.CompositeInstance cimport CompositeInstance


cdef class KnnModel(Model):


    def __init__(self, data: InstanceList, k: int, distanceMetric: DistanceMetric):
        """
        Constructor that sets the data InstanceList, k value and the DistanceMetric.

        PARAMETERS
        ----------
        data : InstanceList
            InstanceList input.
        k : int
            K value.
        distanceMetric : DistanceMetric
            DistanceMetric input.
        """
        self.__data = data
        self.__k = k
        self.__distanceMetric = distanceMetric

    cpdef str predict(self, Instance instance):
        """
        The predict method takes an Instance as an input and finds the nearest neighbors of given instance. Then
        it returns the first possible class label as the predicted class.

        PARAMETERS
        ----------
        instance : Instance
            Instance to make prediction.

        RETURNS
        -------
        str
            The first possible class label as the predicted class.
        """
        cdef InstanceList nearestNeighbors
        cdef str predictedClass
        nearestNeighbors = self.nearestNeighbors(instance)
        if isinstance(instance, CompositeInstance) and nearestNeighbors.size() == 0:
            predictedClass = instance.getPossibleClassLabels()[0]
        else:
            predictedClass = Model.getMaximum(nearestNeighbors.getClassLabels())
        return predictedClass

    def makeComparator(self):
        def compare(instanceA: KnnInstance, instanceB: KnnInstance):
            if instanceA.distance < instanceB.distance:
                return -1
            elif instanceA.distance > instanceB.distance:
                return 1
            else:
                return 0
        return compare

    cpdef InstanceList nearestNeighbors(self, Instance instance):
        """
        The nearestNeighbors method takes an Instance as an input. First it gets the possible class labels, then loops
        through the data InstanceList and creates new list of KnnInstances and adds the corresponding data with
        the distance between data and given instance. After sorting this newly created list, it loops k times and
        returns the first k instances as an InstanceList.

        PARAMETERS
        ----------
        instance : Instance
            Instance to find nearest neighbors

        RETURNS
        -------
        InstanceList
            The first k instances which are nearest to the given instance as an InstanceList.
        """
        cdef InstanceList result
        cdef list instances, possibleClassLabels
        cdef int i
        result = InstanceList()
        instances = []
        possibleClassLabels = []
        if isinstance(instance, CompositeInstance):
            possibleClassLabels = instance.getPossibleClassLabels()
        for i in range(self.__data.size()):
            if not isinstance(instance, CompositeInstance) or self.__data.get(i).getClassLabel() in possibleClassLabels:
                instances.append(KnnInstance(self.__data.get(i), self.__distanceMetric.distance(self.__data.get(i),
                                                                                                instance)))
        instances.sort(key=cmp_to_key(self.makeComparator()))
        for i in range(min(self.__k, len(instances))):
            result.add(instances[i].getInstance())
        return result
