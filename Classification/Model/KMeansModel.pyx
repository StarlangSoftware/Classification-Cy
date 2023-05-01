from Math.DiscreteDistribution cimport DiscreteDistribution

from Classification.DistanceMetric.EuclidianDistance cimport EuclidianDistance

cdef class KMeansModel(GaussianModel):

    cpdef constructor1(self,
                     DiscreteDistribution priorDistribution,
                     InstanceList classMeans,
                     DistanceMetric distanceMetric):
        """
        The constructor that sets the classMeans, priorDistribution and distanceMetric according to given inputs.

        PARAMETERS
        ----------
        priorDistribution : DiscreteDistribution
            DiscreteDistribution input.
        classMeans : InstanceList
            InstanceList of class means.
        distanceMetric : DistanceMetric
            DistanceMetric input.
        """
        self.__class_means = classMeans
        self.prior_distribution = priorDistribution
        self.__distance_metric = distanceMetric

    cpdef constructor2(self, str fileName):
        cdef object inputFile
        self.__distance_metric = EuclidianDistance()
        inputFile = open(fileName, 'r')
        self.loadPriorDistribution(inputFile)
        self.__class_means = self.loadInstanceList(inputFile)
        inputFile.close()

    cpdef InstanceList loadInstanceList(self, object inputFile):
        cdef list types
        cdef int instance_count, i
        cdef InstanceList instance_list
        types = inputFile.readline().strip().split(" ")
        instance_count = int(inputFile.readline().strip())
        instance_list = InstanceList()
        for i in range(instance_count):
            instance_list.add(self.loadInstance(inputFile.readline().strip(), types))
        return instance_list

    def __init__(self,
                 priorDistribution: object,
                 classMeans: InstanceList = None,
                 distanceMetric: DistanceMetric = None):
        if isinstance(priorDistribution, DiscreteDistribution):
            self.constructor1(priorDistribution, classMeans, distanceMetric)
        elif isinstance(priorDistribution, str):
            self.constructor2(priorDistribution)

    cpdef double calculateMetric(self, Instance instance, str Ci):
        """
        The calculateMetric method takes an {@link Instance} and a String as inputs. It loops through the class means,
        if the corresponding class label is same as the given String it returns the negated distance between given
        instance and the current item of class means. Otherwise it returns the smallest negative number.

        PARAMETERS
        ----------
        instance : Instance
            Instance input.
        Ci : str
            String input.

        RETURNS
        -------
        float
            The negated distance between given instance and the current item of class means.
        """
        cdef int i
        for i in range(self.__class_means.size()):
            if self.__class_means.get(i).getClassLabel() == Ci:
                return -self.__distance_metric.distance(instance, self.__class_means.get(i))
        return -1000000
