import copy
from Classification.Parameter.MultiLayerPerceptronParameter cimport MultiLayerPerceptronParameter
from Classification.Performance.ClassificationPerformance cimport ClassificationPerformance
from Classification.InstanceList.InstanceList cimport InstanceList
from Math.Vector cimport Vector


cdef class MultiLayerPerceptronModel(LinearPerceptronModel):

    cpdef __allocateWeights(self, int H, int seed):
        """
        The allocateWeights method allocates layers' weights of Matrix W and V.

        PARAMETERS
        ----------
        H : int
            Integer value for weights.
        """
        self.W = self.allocateLayerWeights(H, self.d + 1, seed)
        self.__V = self.allocateLayerWeights(self.K, H + 1, seed)

    def __init__(self, trainSet: InstanceList, validationSet: InstanceList, parameters: MultiLayerPerceptronParameter):
        """
        A constructor that takes InstanceLists as trainsSet and validationSet. It  sets the NeuralNetworkModel nodes
        with given InstanceList then creates an input vector by using given trainSet and finds error. Via the
        validationSet it finds the classification performance and reassigns the allocated weight Matrix with the matrix
        that has the best accuracy and the Matrix V with the best Vector input.

        PARAMETERS
        ----------
        trainSet : InstanceList
            InstanceList that is used to train.
        validationSet : InstanceList
            InstanceList that is used to validate.
        parameters : MultiLayerPerceptronParameter
            Multi layer perceptron parameters; seed, learningRate, etaDecrease, crossValidationRatio, epoch,
            hiddenNodes.
        """
        cdef Matrix bestW, bestV, deltaV, deltaW
        cdef ClassificationPerformance bestClassificationPerformance, currentClassificationPerformance
        cdef int epoch, i, j
        cdef double learningRate
        cdef Vector hidden, hiddenBiased, rMinusY, oneMinusHidden, tmph, tmpHidden
        self.initWithTrainSet(trainSet)
        self.__allocateWeights(parameters.getHiddenNodes(), parameters.getSeed())
        bestW = copy.deepcopy(self.W)
        bestV = copy.deepcopy(self.__V)
        bestClassificationPerformance = ClassificationPerformance(0.0)
        epoch = parameters.getEpoch()
        learningRate = parameters.getLearningRate()
        for i in range(epoch):
            trainSet.shuffle(parameters.getSeed())
            for j in range(trainSet.size()):
                self.createInputVector(trainSet.get(j))
                hidden = self.calculateHidden(self.x, self.W)
                hiddenBiased = hidden.biased()
                rMinusY = self.calculateRMinusY(trainSet.get(j), hiddenBiased, self.__V)
                deltaV = Matrix(rMinusY, hiddenBiased)
                oneMinusHidden = self.calculateOneMinusHidden(hidden)
                tmph = self.__V.multiplyWithVectorFromLeft(rMinusY)
                tmph.remove(0)
                tmpHidden = oneMinusHidden.elementProduct(hidden.elementProduct(tmph))
                deltaW = Matrix(tmpHidden, self.x)
                deltaV.multiplyWithConstant(learningRate)
                self.__V.add(deltaV)
                deltaW.multiplyWithConstant(learningRate)
                self.W.add(deltaW)
            currentClassificationPerformance = self.testClassifier(validationSet)
            if currentClassificationPerformance.getAccuracy() > bestClassificationPerformance.getAccuracy():
                bestClassificationPerformance = currentClassificationPerformance
                bestW = copy.deepcopy(self.W)
                bestV = copy.deepcopy(self.__V)
            learningRate *= parameters.getEtaDecrease()
        self.W = bestW
        self.__V = bestV

    cpdef calculateOutput(self):
        """
        The calculateOutput method calculates the forward single hidden layer by using Matrices W and V.
        """
        self.calculateForwardSingleHiddenLayer(self.W, self.__V)
