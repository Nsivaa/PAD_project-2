from tqdm import tqdm

class NGram:
    def __init__(self, n, words: tuple):
        self.n = n
        self.words = words
        self.frequency = 0
        self.cohesion = 0
    
    def find_frequency(self, corpus: str):
        """
        Finds the frequency of the n-gram in the corpus and assigns it to the frequency attribute.
        Params:
            corpus (str): The text corpus to analyze.
        """
        # Tokenize the corpus into words
        tokens = corpus.split(" ")
        # Iterate over the words and count the frequency of the n-gram
        for i in tqdm(range(len(tokens) - self.n + 1), mininterval=100):
            n_gram = " ".join(self.words.keys())
            # Check if the n-gram is present in the tokens
            if n_gram == " ".join(tokens[i:i + self.n]):
                self.frequency += 1
                for word in tokens[i:i + self.n]:
                    self.words[word] += 1

    def __str__(self) -> str:
        """
        Returns a string representation of the n-gram object.
        """
        return f"NGram(n = {self.n}, words = {self.words}, frequency={self.frequency}, cohesion={self.cohesion})"
