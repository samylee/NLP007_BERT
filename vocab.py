import tqdm
from collections import Counter


class BertVocab(object):
    def __init__(self, corpus_path, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        specials = ['<pad>', '<unk>', '<eos>', '<sos>', '<mask>']

        # load all words
        counter = self.load_text(corpus_path)
        # frequencies of special tokens are not counted when building vocabulary in frequency order
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        self.itos = list(specials)
        for word, freq in words_and_frequencies:
            if freq < min_freq:
                break
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)
        
    def load_text(self, corpus_path):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            print('Building Vocab')
            counter = Counter()
            for line in tqdm.tqdm(f):
                if isinstance(line, list):
                    words = line
                else:
                    words = line.replace('\n', '').replace('\t', ' ').split()
    
                for word in words:
                    counter[word] += 1
        return counter