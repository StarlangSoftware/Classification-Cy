import unittest

from Classification.Model.Ensemble.BaggingModel import BaggingModel
from Classification.Model.DecisionTree.DecisionTree import DecisionTree
from Classification.Model.DummyModel import DummyModel
from Classification.Model.NonParametric.KnnModel import KnnModel
from Classification.Model.Parametric.LdaModel import LdaModel
from Classification.Model.NeuralNetwork.LinearPerceptronModel import LinearPerceptronModel
from Classification.Model.Parametric.NaiveBayesModel import NaiveBayesModel
from Classification.DistanceMetric.EuclidianDistance import EuclidianDistance
from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.MxKFoldRun import MxKFoldRun
from Classification.Parameter.BaggingParameter import BaggingParameter
from Classification.Parameter.C45Parameter import C45Parameter
from Classification.Parameter.KnnParameter import KnnParameter
from Classification.Parameter.LinearPerceptronParameter import LinearPerceptronParameter
from Classification.Parameter.Parameter import Parameter
from Classification.StatisticalTest.Combined5x2t import Combined5x2t
from test.Classifier.ClassifierTest import ClassifierTest

class Combined5x2tTest(ClassifierTest):

    def test_Compare(self):
        mxKFoldRun = MxKFoldRun(5, 2)
        combined5x2t = Combined5x2t()
        experimentPerformance1 = mxKFoldRun.execute(Experiment(DecisionTree(), C45Parameter(1, True, 0.2), self.iris))
        experimentPerformance2 = mxKFoldRun.execute(Experiment(LinearPerceptronModel(), LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), self.iris))
        self.assertAlmostEqual(0.390, combined5x2t.compare(experimentPerformance1, experimentPerformance2).getPValue(), 3)
        experimentPerformance1 = mxKFoldRun.execute(Experiment(DecisionTree(), C45Parameter(1, True, 0.2), self.tictactoe))
        experimentPerformance2 = mxKFoldRun.execute(Experiment(BaggingModel(), BaggingParameter(1, 50), self.tictactoe))
        self.assertAlmostEqual(0.0000059, combined5x2t.compare(experimentPerformance1, experimentPerformance2).getPValue(), 7)
        experimentPerformance1 = mxKFoldRun.execute(Experiment(LdaModel(), Parameter(1), self.dermatology))
        experimentPerformance2 = mxKFoldRun.execute(Experiment(LinearPerceptronModel(), LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), self.dermatology))
        self.assertAlmostEqual(0.9819, combined5x2t.compare(experimentPerformance1, experimentPerformance2).getPValue(), 4)
        experimentPerformance1 = mxKFoldRun.execute(Experiment(DummyModel(), Parameter(1), self.nursery))
        experimentPerformance2 = mxKFoldRun.execute(Experiment(NaiveBayesModel(), Parameter(1), self.nursery))
        self.assertAlmostEqual(0.0, combined5x2t.compare(experimentPerformance1, experimentPerformance2).getPValue(), 4)
        experimentPerformance1 = mxKFoldRun.execute(Experiment(NaiveBayesModel(), Parameter(1), self.car))
        experimentPerformance2 = mxKFoldRun.execute(Experiment(BaggingModel(), BaggingParameter(1, 50), self.car))
        self.assertAlmostEqual(0.00043, combined5x2t.compare(experimentPerformance1, experimentPerformance2).getPValue(), 5)
        experimentPerformance1 = mxKFoldRun.execute(Experiment(KnnModel(), KnnParameter(1, 3, EuclidianDistance()), self.bupa))
        experimentPerformance2 = mxKFoldRun.execute(Experiment(LdaModel(), Parameter(1), self.bupa))
        self.assertAlmostEqual(0.0663, combined5x2t.compare(experimentPerformance1, experimentPerformance2).getPValue(), 4)


if __name__ == '__main__':
    unittest.main()
