from prepocessing import preprocess
from representation import Glove
import logging
import argparse
import json


DATA_PATH = 'data/'
RESULTS_PATH = 'results/'
LOG_PATH = 'debug.log'


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('path', help='path from DATA_PATH to your txt file')
    argparser.add_argument('-vs', help='size of word-vectors', type=int)
    argparser.add_argument('-i', help='quantity of training loops', type=int)
    args = argparser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.DEBUG,
        filename=LOG_PATH,
        filemode='w',
        datefmt='%H:%M:%S'
    )

    with open(DATA_PATH + args.path) as file:
        tokens = preprocess(file)

    glove = Glove(vector_size=args.vs, iterations=args.i)
    word_vectors = glove.fit(tokens)

    with open(RESULTS_PATH + args.path[:-4] + '.json', 'w') as file:
        json.dump(word_vectors, file)
