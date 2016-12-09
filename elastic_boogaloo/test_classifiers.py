import unittest
import unittest.mock as mock

from .classifiers import *


def fake_distribution(density_value=1.0):
    distribution = mock.Mock()
    distribution.register = mock.Mock()
    distribution.density = mock.Mock(return_value=density_value)

    return distribution


class UnopinionatedBinaryClassifierTests(unittest.TestCase):
    def setUp(self):
        fake_scorer = mock.Mock()
        fake_scorer.score = mock.Mock(return_value=1234.5)

        fake_positive_score_distribution = fake_distribution()

        fake_negative_score_distribution = fake_distribution()

        self.scorer = fake_scorer
        self.positive_score_distribution = fake_positive_score_distribution
        self.negative_score_distribution = fake_negative_score_distribution

        self.classifier = UnopinionatedBinaryClassifier(fake_scorer, fake_positive_score_distribution, fake_negative_score_distribution)

    def test_classify(self):
        observation = 'observation'
        self.assertEqual(self.classifier.classify(observation), 0.5)

    def test_compute_score(self):
        observation = 'mark'
        score = self.classifier.compute_score(observation)

        self.assertEqual(score, 1234.5)
        self.scorer.score.assert_called_once_with(observation)

    def test_train_positive(self):
        observation = 'data'
        self.classifier.train_positive(observation, 1)

        self.scorer.score.assert_called_once_with(observation)
        self.positive_score_distribution.register.assert_called_once_with(1234.5, 1)

    def test_train_negative(self):
        observation = 'data'
        self.classifier.train_negative(observation, 1)

        self.scorer.score.assert_called_once_with(observation)
        self.negative_score_distribution.register.assert_called_once_with(1234.5, 1)
