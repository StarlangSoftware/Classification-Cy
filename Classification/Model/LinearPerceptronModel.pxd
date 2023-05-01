from Math.Matrix cimport Matrix
from Classification.InstanceList.InstanceList cimport InstanceList
from Classification.Model.NeuralNetworkModel cimport NeuralNetworkModel
from Classification.Parameter.LinearPerceptronParameter cimport LinearPerceptronParameter

cdef class LinearPerceptronModel(NeuralNetworkModel):

    cdef Matrix W

    cpdef calculateOutput(self)
    cpdef constructor1(self, InstanceList trainSet)
    cpdef constructor2(self,
                       InstanceList trainSet,
                       InstanceList validationSet,
                       LinearPerceptronParameter parameters)
    cpdef constructor3(self, str fileName)
