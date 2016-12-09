from elastic_boogaloo import classifiers, distributions, scorers
from elasticsearch import Elasticsearch


es_client = Elasticsearch('localhost:9200')
scorer = scorers.ElasticsearchIndexTopScorer(es_client, 'megacorp')

positive_distribution = distributions.ExponentialDistribution()
negative_distribution = distributions.ExponentialDistribution()

classifier = classifiers.UnopinionatedBinaryClassifier(scorer, positive_distribution, negative_distribution)

print('Training douglas as positive...')
classifier.train_positive('douglas')
print('Done')

print('Probability of douglas being positive:', classifier.classify('douglas'))
print('Probability of rock being positive:', classifier.classify('rock'))
