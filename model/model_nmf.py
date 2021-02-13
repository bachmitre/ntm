from sklearn.decomposition import NMF
from model import TopicModel
import pandas as pd


class TopicModelNmf(TopicModel):
    """
    Topic model using NMF
    """
    def __init__(self, vocab=None, n_components=10, n_epochs=10):
        super(TopicModelNmf, self).__init__()
        self.vocab = vocab

        self.model = NMF(
            n_components=n_components,
            alpha=.1,
            l1_ratio=.5,
            max_iter=n_epochs,
            init='nndsvd'
        )

    def fit(self, data):
        self.model.fit(data)

    def transform(self, data):
        return self.model.transform(data)

    def get_topics(self, n_terms=10):
        topics = list()
        for i, topic in enumerate(self.model.components_):
            terms = [self.vocab[i] for i in topic.argsort()[:-n_terms - 1:-1]]
            topics += [terms]
        return pd.DataFrame(topics)
