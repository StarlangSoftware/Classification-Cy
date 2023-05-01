from Math.Matrix cimport Matrix

from Classification.Instance.Instance cimport Instance


cdef class Model(object):

    cpdef str predict(self, Instance instance)
    cpdef dict predictProbability(self, Instance instance)
    cpdef Instance loadInstance(self, str line, list attributeTypes)
    cpdef Matrix loadMatrix(self, object inputFile)
