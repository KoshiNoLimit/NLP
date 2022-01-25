from prepocessing import preprocess
from representetion import Glove
import logging
import argparse


DATA_PATH = 'data/'
RESULTS_PATH = 'results/'


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('path', help='path from DATA_PATH to your txt file')
    argparser.add_argument('-vs', help='size of word-vectors', type=int)
    argparser.add_argument('-i', help='quantity of training loops', type=int)
    args = argparser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.DEBUG,
        filename='debug.log',
        filemode='w',
        datefmt='%H:%M:%S'
    )

    with open(DATA_PATH + args.path) as file:
        tokens = preprocess(file)

    glove = Glove(vector_size=args.vs, iterations=args.i)
    words_ids, word_vectors = glove.fit(tokens)

    with open(RESULTS_PATH + args.path, 'w') as file:
        for word, index in words_ids.items():
            vector = [str(i) for i in word_vectors[index]]
            file.write(' '.join([word] + vector + ['\n']))
