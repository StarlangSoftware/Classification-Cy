import copy

from Classification.InstanceList.Partition cimport Partition
from Math.Vector cimport Vector
from Classification.Parameter.LinearPerceptronParameter cimport LinearPerceptronParameter
from Classification.Performance.ClassificationPerformance cimport ClassificationPerformance

cdef class LinearPerceptronModel(NeuralNetworkModel):
    cpdef constructor1(self, InstanceList trainSet):
        """
        Constructor that sets the NeuralNetworkModel nodes with given InstanceList.
        :param trainSet: InstanceList that is used to train.
        """
        super().__init__(trainSet)

    cpdef constructor2(self,
                       InstanceList trainSet,
                       InstanceList validationSet,
                       LinearPerceptronParameter parameters):
        """
        Constructor that takes InstanceLists as trainsSet and validationSet. Initially it allocates layer weights,
        then creates an input vector by using given trainSet and finds error. Via the validationSet it finds the
        classification performance and at the end it reassigns the allocated weight Matrix with the matrix that has the
        best accuracy.

        PARAMETERS
        ----------
        trainSet : InstanceList
            InstanceList that is used to train.
        validationSet : InstanceList
            InstanceList that is used to validate.
        parameters : LinearPerceptronParameter
            Linear perceptron parameters; learningRate, etaDecrease, crossValidationRatio, epoch.
        """
        cdef Matrix best_w, delta_w
        cdef ClassificationPerformance best_classification_performance, current_classification_performance
        cdef int epoch, i, j
        cdef double learning_rate
        cdef Vector r_minus_y
        self.class_labels = trainSet.getDistinctClassLabels()
        self.K = len(self.class_labels)
        self.d = trainSet.get(0).continuousAttributeSize()
        self.W = self.allocateLayerWeights(row=self.K,
                                           column=self.d + 1,
                                           seed=parameters.getSeed())
        best_w = copy.deepcopy(self.W)
        best_classification_performance = ClassificationPerformance(0.0)
        epoch = parameters.getEpoch()
        learning_rate = parameters.getLearningRate()
        for i in range(epoch):
            trainSet.shuffle(parameters.getSeed())
            for j in range(trainSet.size()):
                self.createInputVector(trainSet.get(j))
                r_minus_y = self.calculateRMinusY(trainSet.get(j), self.x, self.W)
                delta_w = Matrix(r_minus_y, self.x)
                delta_w.multiplyWithConstant(learning_rate)
                self.W.add(delta_w)
            current_classification_performance = self.testClassifier(validationSet)
            if current_classification_performance.getAccuracy() > best_classification_performance.getAccuracy():
                best_classification_performance = current_classification_performance
                best_w = copy.deepcopy(self.W)
            learning_rate *= parameters.getEtaDecrease()
        self.W = best_w

    cpdef constructor3(self, str fileName):
        """
        Loads a linear perceptron model from an input model file.
        :param fileName: Name of the input model file.
        """
        cdef object inputFile
        inputFile = open(fileName, mode='r', encoding='utf-8')
        self.loadClassLabels(inputFile)
        self.W = self.loadMatrix(inputFile)
        inputFile.close()

    cpdef calculateOutput(self):
        """
        The calculateOutput method calculates the Matrix y by multiplying Matrix W with Vector x.
        """
        self.y = self.W.multiplyWithVectorFromRight(self.x)

    cpdef train(self,
                InstanceList trainSet,
                Parameter parameters):
        """
        Training algorithm for the linear perceptron algorithm. 20 percent of the data is separated as cross-validation
        data used for selecting the best weights. 80 percent of the data is used for training the linear perceptron with
        gradient descent.

        PARAMETERS
        ----------
        trainSet : InstanceList
            Training data given to the algorithm
        parameters : LinearPerceptronParameter
            Parameters of the linear perceptron.
        """
        cdef Partition partition
        partition = Partition(instanceList=trainSet,
                              ratio=parameters.getCrossValidationRatio(),
                              seed=parameters.getSeed(),
                              stratified=True)
        self.constructor2(trainSet=partition.get(1),
                          validationSet=partition.get(0),
                          parameters=parameters)

    cpdef loadModel(self, str fileName):
        """
        Loads the linear perceptron model from an input file.
        :param fileName: File name of the linear perceptron model.
        """
        self.constructor3(fileName)
