class ElasticsearchIndexTopScorer:
    """
    A scorer which makes an Elasticsearch query and returns either the top score over all the Elasticsearch hits or
    returns 0
    """
    def __init__(self, es_client, index):
        """
        :param es_client: Elasticsearch server against which to make queries
        :param index: An index against which to make search queries
        """
        self.es = es_client
        self.index = index

    def score(self, observed_string):
        """
        Returns top Elasticsearch hits score for a basic search query against the scorer's index
        :param observed_string: The observation that the caller wishes to score
        :return: The top score of any hit for the observed_string against the scorer's index
        """
        res = self.es.search(index=self.index, body={'query': {'match': {'_all': observed_string}}})
        hits = res['hits']['hits']
        if len(hits) > 0:
            score_string = hits[0]['_score']
            score = float(score_string)
            return score
        return 0.0
