from Classification.InstanceList.InstanceList cimport InstanceList
from Classification.Model.NeuralNetworkModel cimport NeuralNetworkModel
from Classification.Parameter.DeepNetworkParameter cimport DeepNetworkParameter


cdef class DeepNetworkModel(NeuralNetworkModel):

    cdef list __weights
    cdef int __hidden_layer_size
    cdef object __activation_function

    cpdef __allocateWeights(self, DeepNetworkParameter parameters)
    cpdef list __setBestWeights(self)
    cpdef calculateOutput(self)
    cpdef constructor1(self,
                       InstanceList trainSet,
                       InstanceList validationSet,
                       DeepNetworkParameter parameters)
    cpdef constructor2(self, str fileName)
