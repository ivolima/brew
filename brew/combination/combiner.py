import numpy as np
from brew.combination import rules


class Combiner(object):

    __VALID_WEIGHTED_COMBINATION_RULES = [
        rules.majority_vote_rule,
        rules.mean_rule,
    ]

    def __init__(self, rule='majority_vote'):
        self.combination_rule = rule

        if rule == 'majority_vote':
            self.rule = rules.majority_vote_rule

        elif rule == 'max':
            self.rule = rules.max_rule

        elif rule == 'min':
            self.rule = rules.min_rule

        elif rule == 'mean':
            self.rule = rules.mean_rule

        elif rule == 'median':
            self.rule = rules.median_rule

        else:
            raise Exception('invalid argument rule for Combiner class')

    def combine(self, results, weights=None):
        """
        This method puts together the results of all classifiers
        based on a pre-selected combination rule.

        Parameters
        ----------
        results: array-like, shape = [n_samples, n_classes, n_classifiers]
                    If combination rule is 'majority_vote' results should be Ensemble.output(X, mode='votes')
                    Otherwise, Ensemble.output(X, mode='probs')
        weights: array-like, optional(default=None)
                    Weights of the classifiers. Must have the same size of n_classifiers.
                    Applies only to 'majority_vote' and 'mean' combination rules.
        """

        nresults = results.copy().astype(float)
        n_samples = nresults.shape[0]
        y_pred = np.zeros((n_samples,))

        if weights is not None:
            # verify valid combination rules
            if self.rule in __VALID_WEIGHTED_COMBINATION_RULES:
                # verify shapes
                if weights.shape[0] != nresults.shape[2]:
                    raise Exception(
                        'weights and classifiers must have same size')

                # apply weights
                for i in range(nresults.shape[2]):
                    nresults[:, :, i] = nresults[:, :, i] * weights[i]

        for i in range(n_samples):
            y_pred[i] = self.rule(nresults[i, :, :])

        return y_pred
