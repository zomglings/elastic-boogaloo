class UnopinionatedBinaryClassifier:
    """
    Classifies data using the search score produced by a query against an ElasticSearch index.

    Can be trained.

    Assumes no particular parametrization of the class distribution, and is provided a parametrization of the
    distributions of observations given class and not class.

    Uses a simple likelihood-based calculation to produce a classification score.
    """

    def __init__(self, scorer, positive_score_distribution, negative_score_distribution):
        """
        :param scorer: An object with a score method, which accepts observations and assigns scores to them which are
        consistent with the values produced by sampling from the positive_score_distribution and
        negative_score_distribution
        :param positive_score_distribution: The distribution to be used to explain the likelihood of ElasticSearch
        scores given the positive case, assumed to be of type distributions.Distribution
        :param negative_score_distribution: The distribution to be used to explain the likelihood of ElasticSearch
        scores given the negative case, assumed to be of type distributions.Distribution
        """
        self.scorer = scorer
        self.positive_score_distribution = positive_score_distribution
        self.negative_score_distribution = negative_score_distribution

    def classify(self, observation):
        """
        Estimates how likely the given observation is to belong to the positive class. The assumption is that, if it
        does not belong to the positive class, then it belongs to the negative one.
        :param value: An observation
        :return: The probability that the given observation belongs to the positive class
        """
        score = self.compute_score(observation)

        positive_score_density = self.positive_score_distribution.density(score)
        negative_score_density = self.negative_score_distribution.density(score)

        positive_probability = positive_score_density/(positive_score_density + negative_score_density)

        return positive_probability

    def train_positive(self, observation, confidence=1.0):
        """
        Trains the classifier with an observation from the positive class
        :param observation: An observation that is suspected to be positive
        :param confidence: The probability with which the given observation is known to be positive
        :return: None
        """
        score = self.compute_score(observation)
        self.positive_score_distribution.register(score, confidence)

    def train_negative(self, observation, confidence=1.0):
        """
        Trains the classifier with an observation from the negative class
        :param observation: An observation that is suspected to be negative
        :param confidence: The probability with which the given observation is known to be negative
        :return: None
        """
        score = self.compute_score(observation)
        self.negative_score_distribution.register(score, confidence)

    def compute_score(self, observation):
        """
        Scores the given observation against the classifier's scorer.
        :param observation: An observation we wish to score
        :return: The value the scorer assigns to the input observation
        """
        return self.scorer.score(observation)
