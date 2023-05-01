from Math.DiscreteDistribution cimport DiscreteDistribution

from Classification.Instance.Instance cimport Instance
from Classification.Model.LdaModel cimport LdaModel

cdef class QdaModel(LdaModel):

    cdef dict __W

    cpdef double calculateMetric(self, Instance instance, str Ci)
    cpdef constructor3(self,
                     DiscreteDistribution priorDistribution,
                     dict W,
                     dict w,
                     dict w0)
    cpdef constructor2(self, str fileName)
