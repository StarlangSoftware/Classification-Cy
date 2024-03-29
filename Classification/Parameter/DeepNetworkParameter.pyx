cdef class DeepNetworkParameter(LinearPerceptronParameter):

    def __init__(self,
                 seed: int,
                 learningRate: float,
                 etaDecrease: float,
                 crossValidationRatio: float,
                 epoch: int,
                 hiddenLayers: list,
                 activationFunction: object):
        """
        Parameters of the deep network classifier.

        PARAMETERS
        ----------
        seed : int
            Seed is used for random number generation.
        learningRate : float
            Double value for learning rate of the algorithm.
        etaDecrease : float
            Double value for decrease in eta of the algorithm.
        crossValidationRatio : float
            Double value for cross validation ratio of the algorithm.
        epoch : int
            Integer value for epoch number of the algorithm.
        hiddenLayers : list
            An integer list for hidden layers of the algorithm.
        activationFunction : ActivationFunction
            Activation function.
        """
        super().__init__(seed, learningRate, etaDecrease, crossValidationRatio, epoch)
        self.__hiddenLayers = hiddenLayers
        self.__activationFunction = activationFunction

    cpdef int layerSize(self):
        """
        The layerSize method returns the size of the hiddenLayers list.

        RETURNS
        -------
        int
            The size of the hiddenLayers {@link ArrayList}.
        """
        return len(self.__hiddenLayers)

    cpdef int getHiddenNodes(self, int layerIndex):
        """
        The getHiddenNodes method takes a layer index as an input and returns the element at the given index of hiddenLayers
        list.

        PARAMETERS
        ----------
        layerIndex : int
            Index of the layer.

        RETURNS
        -------
        int
            The element at the layerIndex of hiddenLayers list.
        """
        return self.__hiddenLayers[layerIndex]

    cpdef object getActivationFunction(self):
        """
        Accessor for the activation function.

        RETURNS
        -------
        ActivationFunction
            The activation function.
        """
        return self.__activationFunction
