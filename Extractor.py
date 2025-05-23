from NGramData import NGramData
from tqdm import tqdm
from collections import defaultdict
from enum import Enum, auto


class Extractor:
    """
    The Extractor class is responsible for extracting n-grams from a given corpus.
    Stores n-grams in a defaultdict collection, as it's more concise to count elements with it (easier to add 
    non-existing keys). The dictionary is formed by tuples of tokens as keys (the words forming the n-gram), 
    and the corresponding n-gram object as value. In this way we can access frequency and all other metadata of
    the n-gram in O(1) by dictionary look-up on the words tuple.
    """

    def __init__(self, corpus, n_max: int = 7, limit: int = None):
        
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.n_max = n_max
        self.n_grams = defaultdict(NGramData)
        self.find_n_grams(limit) 
        self.total_ngrams = defaultdict(int) # to count the total number of n-grams of each size
        self.find_total_ngrams()

    class GluesEnum(Enum):
        scp = auto()
        dice = auto()
        phi_square = auto()

    def find_total_ngrams(self):
        """
        Finds the total number of n-grams of each size in the corpus and stores them in the total_ngrams attribute.
        """
        for ngram, data in self.n_grams.items():
            self.total_ngrams[len(ngram)] += data.frequency

    def find_frequency(self, ngram_words):
        data = self.n_grams.get(ngram_words)
        return data.frequency if data else 0

    def find_probability(self, ngram_words):
        data = self.n_grams.get(ngram_words)
        if not data:
            return 0
        total = self.total_ngrams[len(ngram_words)]
        return data.frequency / sum(self.total_ngrams.values())

    def calculate_scp(self, ngram_words, ngram_data):
        n = len(ngram_words)
        if n == 2:
            p1 = self.find_probability((ngram_words[0],))
            p2 = self.find_probability((ngram_words[1],))
            p12 = self.find_probability(ngram_words)
            denom = p1 * p2
            if denom > 0:
                ngram_data.scp = (p12 ** 2) / denom
            return

        # For n > 2
        F = 0
        for i in range(1, n): # range (1, n) goes up to n - 1
            p_left = self.find_probability(ngram_words[:i])
            p_right = self.find_probability(ngram_words[i:])
            F += p_left * p_right
        F /= (n - 1)
        p_full = self.find_probability(ngram_words)
        if F > 0:
            ngram_data.scp = (p_full ** 2) / F


    def calculate_dice(self, ngram_words, ngram_data):
        pass

    def calculate_phi_square(self, ngram_words, ngram_data):
        pass

    GLUE_FUNCTIONS = {
        GluesEnum.scp : calculate_scp,
        GluesEnum.dice: calculate_dice,
        GluesEnum.phi_square: calculate_phi_square
    }

    
    def find_glue_values(self):
        """
        finds the glue values of the n-gram in the corpus for all the glue functions and assigns it to the cohesion attribute.
        """
        # for each glue function
        for glue in tqdm(self.GLUE_FUNCTIONS.keys(), desc="Finding glue values", unit="glue"):
            # for each ngram
            for words in list(self.n_grams.keys()):
                data = self.n_grams[words]
                # if not an unigram
                if len(words) > 1:
                    # call the glue function
                    self.GLUE_FUNCTIONS[glue](self, words, data)
            break
        else:
            raise ValueError(f"Invalid glue type: {type}")


    def find_n_grams(self, limit):
        """
        Finds n-grams in the corpus and stores them in the n_grams attribute, 
        while also counting their frequencies, for every n from 2 to n_max.
        """
        # for every size of the n-grams, up to n_max. also stores unigrams
        for i in tqdm(range(1, self.n_max + 1), desc=f"Finding n-grams"): 
            range_limit = len(self.corpus) - i + 1 if not limit else limit
            for j in tqdm(range(0, range_limit), mininterval=50000, unit="n-gram", desc=f"Finding n-grams of size {i} in corpus"):
            # create an n-gram of size i and store it in the dictionary if it doesn't exist, else increase its frequency by 1
                words = tuple(self.corpus[j:j + i])
                self.n_grams[words].frequency += 1
    

    def sort_by_glue(self, glue: GluesEnum = GluesEnum.scp):
        """
        Sorts the n-grams by the glue function specified in the glue parameter.
        """
        if glue not in self.GLUE_FUNCTIONS:
            raise ValueError(f"Invalid glue type: {glue}")
        # sort the n-grams by the glue function
        self.n_grams = dict(sorted(self.n_grams.items(), key=lambda item: getattr(item[1], glue.name), reverse=True))
    
    def __str__(self) -> str:
        """
        Returns a string representation of the extractor object: 
        """

        return str([f"{str(words)} : {str(data)}" for (words, data) in self.n_grams.items()])

    def print_to_file(self, file_path: str = "n_grams.txt"):
        """
        Prints the n-grams to a file in a readable format.
        """
        with open(file_path, "w") as f:
            print(self, file=f)
    
    def print_top_n_glue(self, n: int = 10, glue: GluesEnum = GluesEnum.scp):
        """
        Prints the top n n-grams by the glue function specified in the glue parameter.
        """
        if glue not in self.GLUE_FUNCTIONS:
            raise ValueError(f"Invalid glue type: {glue}")
        # sort the n-grams by the glue function
        self.sort_by_glue(glue)
        # print the top n n-grams
        for i, (words, data) in enumerate(list(self.n_grams.items())[:n]):
            print(f"{i + 1}: {words} : {data}")
    
    def print_bottom_n_glue(self, n: int = 10, glue: GluesEnum = GluesEnum.scp):
        """
        Prints the bottom n n-grams by the glue function specified in the glue parameter.
        """
        if glue not in self.GLUE_FUNCTIONS:
            raise ValueError(f"Invalid glue type: {glue}")
        # sort the n-grams by the glue function
        self.sort_by_glue(glue)
        # print the bottom n n-grams
        for i, (words, data) in enumerate(list(self.n_grams.items())[-n:]):
            print(f"{i + 1}: {words} : {data}")