from Classification.Classifier.Classifier cimport Classifier
from Classification.InstanceList.InstanceList cimport InstanceList
from Classification.Model.KnnModel cimport KnnModel
from Classification.Parameter.Parameter cimport Parameter


cdef class Knn(Classifier):

    cpdef train(self,
                InstanceList trainSet,
                Parameter parameters):
        """
        Training algorithm for K-nearest neighbor classifier.

        PARAMETERS
        ----------
        trainSet : InstanceList
            Training data given to the algorithm.
        parameters : KnnParameter
            Parameters of the Knn algorithm.
        """
        self.model = KnnModel(data=trainSet,
                              k=parameters.getK(),
                              distanceMetric=parameters.getDistanceMetric())

    cpdef loadModel(self, str fileName):
        """
        Loads the K-nearest neighbor model from an input file.
        :param fileName: File name of the K-nearest neighbor model.
        """
        self.model = KnnModel(fileName)
