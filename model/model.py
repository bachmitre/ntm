from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from collections import Counter
import numpy as np


class TopicModel(object):
    def __init__(self, vocab=None, n_components=10, n_epochs=100):
        self.n_components = n_components

    def fit(self, data):
        pass

    def transform(self, data):
        return None

    def get_topics(self, n_terms=10):
        return None

    def get_coherence(self, docs=None, dictionary=None, corpus=None, n_terms=10):

        topics = self.get_topics(n_terms=n_terms)

        if not dictionary and not corpus:
            dictionary = Dictionary(docs)
            corpus = [dictionary.doc2bow(t) for t in docs]

        return CoherenceModel(
            topn=self.n_components,
            texts=docs,
            topics=topics.values,
            corpus=corpus,
            dictionary=dictionary,
            coherence='c_npmi'
        ).get_coherence()

    def get_topic_uniqueness(self, n_terms=10):
        """
        https://github.com/awslabs/w-lda/blob/master/utils.py
        This function calculates topic uniqueness scores for a given list of topics.
        For each topic, the uniqueness is calculated as:  (\sum_{i=1}^n 1/cnt(i)) / n,
        where n is the number of top words in the topic and cnt(i) is the counter for the number of times the word
        appears in the top words of all the topics.
        """
        top_words_idx_all_topics = self.get_topics(n_terms=n_terms).values
        n_topics = len(top_words_idx_all_topics)

        # build word_cnt_dict: number of times the word appears in top words
        word_cnt_dict = Counter()
        for i in range(n_topics):
            word_cnt_dict.update(top_words_idx_all_topics[i])

        uniqueness_dict = dict()
        for i in range(n_topics):
            cnt_inv_sum = 0.0
            for ind in top_words_idx_all_topics[i]:
                cnt_inv_sum += 1.0 / word_cnt_dict[ind]
            uniqueness_dict[i] = cnt_inv_sum / len(top_words_idx_all_topics[i])

        return np.mean(list(uniqueness_dict.values()))

    def get_topic_diversity(self, n_terms=10):
        """
        Topic diversity adapted from
        paper: https://arxiv.org/abs/1907.04907
        code: https://github.com/adjidieng/ETM/blob/master/utils.py
        """
        topics = self.get_topics(n_terms=n_terms).values
        num_topics = len(topics)
        n_unique = len(np.unique(topics))
        topic_diversity = n_unique / (n_terms * num_topics)
        return topic_diversity
