import unittest
import unittest.mock as mock

from .scorers import *


class ElasticsearchIndexTopScorerTests(unittest.TestCase):
    def setUp(self):
        fake_score_string = '73.21'
        fake_score = 73.21

        fake_es_client = mock.Mock()
        fake_es_client.search = mock.Mock(return_value={'hits': {'hits': [{'_score': fake_score_string}]}})

        fake_index = 'fake_index'

        self.fake_score = fake_score
        self.fake_es_client = fake_es_client
        self.fake_index = fake_index

        self.scorer = ElasticsearchIndexTopScorer(fake_es_client, 'fake_index')

    def test_score(self):
        observed_string = 'i am a little teapot'
        score = self.scorer.score(observed_string)

        self.assertEqual(score, self.fake_score)
        self.fake_es_client.search.assert_called_once_with(index=self.fake_index, body={'query':{'match': {'_all': observed_string}}})
