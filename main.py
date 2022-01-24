from prepocessing import preprocess
from representetion import Glove


path = "HP3.txt"


with open('data/' + path) as file:
    tokens = preprocess(file)

print('glove starts')

glove = Glove(vector_size=100, iterations=200)
words_ids, word_vectors = glove.fit(tokens)

with open('results/' + path, 'w') as file:
    for word, index in words_ids.items():
        vector = [str(i) for i in word_vectors[index]]
        file.write(' '.join([word] + vector + ['\n']))
