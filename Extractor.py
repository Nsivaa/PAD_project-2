from NGram import NGram
from tqdm import tqdm

class Extractor:
    """
    The Extractor class is responsible for extracting n-grams from a given corpus.
    """

    def __init__(self, corpus, n_max: int = 7):
        self.corpus = corpus
        self.n_max = n_max
        self.n_grams = []
        self.find_n_grams() # n-grams sorted by cohesion

    def find_n_grams(self):
        """
        Finds n-grams in the corpus and stores them in the n_grams attribute as a tensor, for every n from 2 to n_max.
        """
        for i in tqdm(range(2, self.n_max + 1), desc=f"Finding n-grams"):
            #for j in tqdm(range(0, len(self.corpus) - i + 1), mininterval=50000, unit="n-gram", desc=f"Finding n-grams of size {i} in corpus"):
            for j in tqdm(range(0, 50), mininterval=50000, unit="n-grams", desc=f"Finding n-grams of size {i} in corpus"):

                n_gram = NGram(n = i, words =  tuple(self.corpus[j:j + i]))
                if n_gram not in self.n_grams:
                    self.n_grams.append(n_gram)

    def find_n_grams_frequencies(self):
        """
        Generates n-gram objects with n from 2 to n_max and finds their frequencies in the corpus.
        """
        for n_gram in tqdm(self.n_grams, unit="n-grams", desc=f"Finding n-grams frequencies"):
            n_gram.find_frequency(self.corpus)

    def __str__(self) -> str:
        """
        Returns a string representation of the extractor object.
        """

        return str([str(n_gram) for n_gram in self.n_grams])