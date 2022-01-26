import json
import argparse
import numpy as np


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('path', help='path from RESULTS_PATH to your json word:vectors file')
    argparser.add_argument('word', help='path from DATA_PATH to your txt file')
    argparser.add_argument('-cnt', help='quantity synonyms', type=int, default=5)
    args = argparser.parse_args()

    with open('results/'+ args.path) as file:
        vocabulary = json.load(file)
    self = np.array(vocabulary.pop(args.word))

    synonyms = [(word, np.linalg.norm(self-np.array(vector))) for word, vector in vocabulary.items()]
    synonyms.sort(key=lambda x: x[1])

    for i in range(args.cnt):
        print(synonyms[i])
