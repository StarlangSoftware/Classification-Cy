from Classification.Instance.Instance cimport Instance
from Classification.Model.Model cimport Model


cdef class RandomModel(Model):

    cdef list __class_labels
    cdef int __seed

    cpdef str predict(self, Instance instance)
    cpdef dict predictProbability(self, Instance instance)
    cpdef constructor1(self, list classLabels, int seed)
    cpdef constructor2(self, str fileName)
