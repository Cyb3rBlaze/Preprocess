import tensorflow as tf
from tqdm import tqdm
import re
import numpy as np
from collections import Counter
import os

#####################
#PREPROCESS DATA
#####################
def preprocess_data():
    with open('./data/lyrics.csv') as csv_file:
        labels = []
        train = []

        allData = ""
        songs = []

        print("Filtering data phase 1...")
        for i in tqdm(csv_file):
            allData += i

        print("Filtering data phase 2...")
        start = False
        curr_song = ""
        for i in tqdm(allData):
            if start == True:
                if i == '"':
                    start = False
                    songs += [curr_song]
                else:
                    curr_song += i
            elif start == False and i == '"':
                start = True
                curr_song = ""

        print("Filtering data phase 3...")
        train = []
        for song in tqdm(songs):
            filtered = re.sub('[.?!W#@,]', '', song)
            filtered = re.sub('\n', ' ', filtered)
            filtered = filtered.lower()
            allData += filtered + " "
            train += [filtered.split(" ")]

        allWords = allData.split(" ")
        del allData
        counter = Counter(allWords)
        most_occur = counter.most_common(400000)
        del counter

        print("Filtering data phase 4...")
        vocab = [set[0] for set in most_occur]
        word2idx = {word:idx for idx, word in enumerate(vocab)}
        idx2word = {idx:word for idx, word in enumerate(vocab)}

        '''with open('word_Id.p', 'wb') as fp:
            pickle.dump(word2idx, fp, protocol=pickle.HIGHEST_PROTOCOL)'''

        del vocab
        del most_occur

        print("Filtering data phase 5...")
        vectors = []
        words = []
        for i in tqdm(range(len(allWords))):
            words += allWords[i]
        for i in tqdm(range(len(words))):
            try:
                vectors += [word2idx[words[i]]]
            except:
                vectors += [400001]

        del allWords
        del word2idx
        del idx2word
        del train

        print("Filtering data phase 6...")
        data = []
        for i in tqdm(range(len(vectors))):
            if i < len(vectors)-1:
                data += [(vectors[i], vectors[i+1])]
            else:
                pass

        del vectors
        del labels

        np.random.shuffle(data)

        print("Filtering data phase 7...")
        vectors = []
        labels = []
        for i in tqdm(data):
            vectors += [i[0]]
            labels += [i[1]]

        del data

        return vectors, labels

data = preprocess_data()

print(data[:10])
