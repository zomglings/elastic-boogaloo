import unittest

from distributions import *


class ExponentialDistributionTest(unittest.TestCase):
    def setUp(self):
        self.distribution = ExponentialDistribution(1)

    def test_register(self):
        distribution = self.distribution
        distribution.register(2, 1)
        self.assertEqual(distribution.observation_mean, 2)
        self.assertEqual(distribution.observations, 1)
        self.assertEqual(distribution.rate, 1/2)

    def test_density(self):
        distribution = self.distribution
        self.assertEqual(distribution.rate, 1)
        self.assertEqual(distribution.density(1), 1/np.e)
