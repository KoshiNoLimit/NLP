import numpy as np
from random import shuffle
import itertools


class Glove:

    def __init__(self, window=5, vector_size=100, iterations=25, x_max=20, alpha=0.75, learning_rate=0.05):
        self.window = window
        self.vector_size = vector_size
        self.iterations = iterations
        self.weight = lambda x: 1 if x > x_max else (x / x_max) ** alpha
        self.learning_rate = learning_rate

    def fit(self, tokens):
        vocabulary, co_occurrence = self.build_co_occurrence(tokens)

        W = (np.random.rand(len(vocabulary) * 2, self.vector_size) - 0.5) / float(self.vector_size+1)
        biases = (np.random.rand(len(vocabulary) * 2) - 0.5) / float(self.vector_size + 1)
        gradient_squared = np.ones((len(vocabulary) * 2, self.vector_size), dtype=np.float64)
        gradient_squared_biases = np.ones(len(vocabulary) * 2, dtype=np.float64)

        data = [(
            W[i_main],
            W[i_context + len(vocabulary)],
            biases[i_main: i_main + 1],
            biases[i_context + len(vocabulary): i_context + len(vocabulary) + 1],
            gradient_squared[i_main], gradient_squared[i_context + len(vocabulary)],
            gradient_squared_biases[i_main: i_main + 1],
            gradient_squared_biases[i_context + len(vocabulary): i_context + len(vocabulary) + 1],
            co_occurrence[i_main, i_context]
        ) for i_main, i_context in itertools.permutations(range(len(co_occurrence)), 2) if co_occurrence[i_main, i_context] > 0]

        for i in np.arange(self.iterations):
            cost = self.iter(data)
            print('Iteration {}.   cost={}'.format(i, cost))

        return vocabulary, W

    def iter(self, data):
        global_cost = 0

        shuffle(data)

        for (v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context, gradsq_b_main, gradsq_b_context,
             cooccurrence) in data:
            weight = self.weight(cooccurrence)

            cost_inner = (np.dot(v_main, v_context) + b_main[0] + b_context[0] - np.log(cooccurrence))
            cost = weight * cost_inner**2
            global_cost += 0.5 * cost

            grad_main = weight * cost_inner * v_context
            grad_context = weight * cost_inner * v_main

            grad_bias_main = weight * cost_inner
            grad_bias_context = weight * cost_inner

            v_main -= (self.learning_rate * grad_main / np.sqrt(gradsq_W_main))
            v_context -= (self.learning_rate * grad_context / np.sqrt(gradsq_W_context))

            b_main -= (self.learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
            b_context -= (self.learning_rate * grad_bias_context / np.sqrt(gradsq_b_context))

            gradsq_W_main += np.square(grad_main)
            gradsq_W_context += np.square(grad_context)
            gradsq_b_main += grad_bias_main ** 2
            gradsq_b_context += grad_bias_context ** 2

        return global_cost

    def build_co_occurrence(self, tokens):
        words = {word: i for i, word in enumerate(set(tokens))}
        co_occurrence = np.zeros([len(words), len(words)])

        for i in np.arange(len(tokens) - self.window):
            for j in np.arange(i + 1, i + self.window):
                co_occurrence[words[tokens[i]], words[tokens[j]]] += 1
                co_occurrence[words[tokens[j]], words[tokens[i]]] += 1

        return words, co_occurrence
