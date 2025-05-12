from NGram import NGram
from tqdm import tqdm
from collections import defaultdict
class Extractor:
    """
    The Extractor class is responsible for extracting n-grams from a given corpus.
    Stores n-grams in a defaultdict collection, as it's more concise to count elements with it (easier to add 
    non-existing keys). The dictionary is formed by tuples of tokens as keys (the words forming the n-gram), 
    and the corresponding n-gram object as value. In this way we can access frequency and all other metadata of
    the n-gram in O(1) by dictionary look-up on the words tuple.
    """

    def __init__(self, corpus, n_max: int = 7):
        self.corpus = corpus
        self.n_max = n_max
        self.n_grams = defaultdict(NGram)
        self.find_n_grams() 

    def find_n_grams(self):
        """
        Finds n-grams in the corpus and stores them in the n_grams attribute as a tensor, for every n from 2 to n_max.
        """
        # for every size of the n-grams, up to n_max. also stores unigrams
        for i in tqdm(range(1, self.n_max + 1), desc=f"Finding n-grams"): 
            # for j in tqdm(range(0, len(self.corpus) - i + 1), mininterval=50000, unit="n-gram", desc=f"Finding n-grams of size {i} in corpus"):
            # create an n-gram of size i and store it in the dictionary if it doesn't exist, else increase its frequency by 1
            for j in tqdm(range(0, 50000), mininterval=50000, unit="n-grams", desc=f"Finding n-grams of size {i} in corpus"):
                words = tuple(self.corpus[j:j + i])
                self.n_grams[words].frequency += 1


    def find_n_grams_frequencies(self):
        """
        Generates n-gram objects with n from 2 to n_max and finds their frequencies in the corpus.
        """
        for n_gram in tqdm(self.n_grams, unit="n-grams", desc=f"Finding n-grams frequencies"):
            n_gram.find_frequency(self.corpus)

    def __str__(self) -> str:
        """
        Returns a string representation of the extractor object: 
        """

        return str([str(words) + " : " + str(data) for (words, data) in self.n_grams.items()])

    def print_to_file(self, file_path: str = "n_grams.txt"):
        """
        Prints the n-grams to a file in a readable format.
        """
        with open(file_path, "w") as f:
            print(self, file=f)