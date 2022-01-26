import numpy as np
from random import shuffle
import itertools
import logging
from tqdm import tqdm


class Glove:

    VECTOR_SIZE = 100
    ITERATIONS = 100
    WINDOW = 5
    LEARNING_RATE = 0.05
    X_MAX = 20
    ALPHA = 0.75

    def __init__(self, vector_size=None, iterations=None, window=None, learning_rate=None, x_max=None, alpha=None):
        self.vector_size = vector_size if vector_size is not None else Glove.VECTOR_SIZE
        self.iterations = iterations if iterations is not None else Glove.ITERATIONS
        self.window = window if window is not None else Glove.WINDOW
        self.learning_rate = learning_rate if learning_rate is not None else Glove.LEARNING_RATE

        x_max = x_max if x_max is not None else Glove.X_MAX
        alpha = alpha if alpha is not None else Glove.ALPHA
        self.weight = lambda x: 1 if x > x_max else (x / x_max) ** alpha

        logging.debug(f'Glove: {self.__dict__}')

    def fit(self, tokens):
        logging.debug('fitting started')
        vocabulary, co_occurrence = self.build_co_occurrence(tokens)
        logging.debug('co-occurrence matrix is obtained')

        W = (np.random.rand(len(vocabulary) * 2, self.vector_size) - 0.5) / float(self.vector_size+1)
        biases = (np.random.rand(len(vocabulary) * 2) - 0.5) / float(self.vector_size + 1)
        gradient_squared = np.ones((len(vocabulary) * 2, self.vector_size), dtype=np.float64)
        gradient_squared_biases = np.ones(len(vocabulary) * 2, dtype=np.float64)

        data = [(
            W[i],
            W[j + len(vocabulary)],
            biases[i: i + 1],
            biases[j + len(vocabulary): j + len(vocabulary) + 1],
            gradient_squared[i], gradient_squared[j + len(vocabulary)],
            gradient_squared_biases[i: i + 1],
            gradient_squared_biases[j + len(vocabulary): j + len(vocabulary) + 1],
            co_occurrence[i, j]
        ) for i, j in itertools.permutations(range(len(co_occurrence)), 2) if co_occurrence[i, j] > 0]

        for i in tqdm(np.arange(self.iterations),
                      total=self.iterations, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}'):
            cost = self.train_vectors(data)
            logging.info(f' iteration {i}    cost={cost:.4f}')

        return {word: (W[i]+W[i+len(vocabulary)]).tolist() for word, i in vocabulary.items()}

    def train_vectors(self, data):
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
