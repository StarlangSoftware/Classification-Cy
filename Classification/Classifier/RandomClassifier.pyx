from Classification.Classifier.Classifier cimport Classifier
from Classification.InstanceList.InstanceList cimport InstanceList
from Classification.Model.RandomModel cimport RandomModel
from Classification.Parameter.Parameter cimport Parameter


cdef class RandomClassifier(Classifier):

    cpdef train(self,
                InstanceList trainSet,
                Parameter parameters):
        """
        Training algorithm for random classifier.

        PARAMETERS
        ----------
        trainSet : InstanceList
            Training data given to the algorithm.
        """
        self.model = RandomModel(classLabels=list(trainSet.classDistribution().keys()),
                                 seed=parameters.getSeed())

    cpdef loadModel(self, str fileName):
        """
        Loads the random classifier model from an input file.
        :param fileName: File name of the random classifier model.
        """
        self.model = RandomModel(fileName)
