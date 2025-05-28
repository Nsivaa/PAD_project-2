from NGramData import NGramData
from tqdm import tqdm
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        self.stopwords = set()
        self.glue_functions = {
            "scp": self.calculate_scp,
            "dice": self.calculate_dice,
            "phi_square": self.calculate_phi_square
        }
        self.glue_type = "scp"  # default glue
        self.p = 2  # default p value for glue functions, can be changed later

    def get_glue_value(self, data: NGramData):
        return {
            "scp": data.scp,
            "dice": data.dice,
            "phi_square": data.phi_square
        }[self.glue_type]



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
            return 0.0
        n = len(ngram_words)
        total = self.total_ngrams[n]
        return data.frequency / total if total > 0 else 0.0


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
        """         
        n = len(ngram_words)
        if n == 2:
            p1 = self.find_frequency((ngram_words[0],))
            p2 = self.find_frequency((ngram_words[1],))
            p12 = self.find_frequency(ngram_words)
            denom = p1 + p2
            if denom > 0:
                ngram_data.dice = (p12 * 2) / denom
            return

        # For n > 2
        F = 0
        for i in range(1, n): # range (1, n) goes up to n - 1
            p_left = self.find_frequency(ngram_words[:i])
            p_right = self.find_frequency(ngram_words[i:])
            F += p_left + p_right
        F /= (n - 1)
        p_full = self.find_frequency(ngram_words)
        if F > 0:
            ngram_data.dice = (p_full * 2) / F 
        """

    def calculate_phi_square(self, ngram_words, ngram_data):
        pass
        """ n = len(ngram_words)
        N = self.corpus_size
        f_full = self.find_frequency(ngram_words)
        ngram_data.phi_square = 0
        if f_full == 0:
            return

        phi_values = []
        for i in range(1, n):
            f_left = self.find_frequency(ngram_words[:i])
            f_right = self.find_frequency(ngram_words[i:])
            expected = (f_left * f_right) / N if N > 0 else 0
            if expected > 0:
                numerator = (f_full - expected) ** 2
                phi = numerator / expected
                phi_values.append(phi)
        
        if phi_values:
            ngram_data.phi_square = sum(phi_values) / len(phi_values) """


    ####
    def filter_by_min_frequency(self, min_freq: int):
        """
            Filters out n-grams with frequency below min_freq.
        """
        self.n_grams = {ngram: data for ngram, data in self.n_grams.items() if data.frequency >= min_freq}


    
    def find_glue_values(self):
        """
        Computes and assigns glue values for all n-grams (of length > 1) using all defined glue functions.
        Each result is saved in the corresponding attribute of the NGramData object.
        """
        for glue_name, glue_func in tqdm(self.glue_functions.items(), desc="Finding glue values", unit="glue"):
            for words, data in self.n_grams.items():
                if len(words) > 1:
                    glue_func(words, data)


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
    

    def sort_by_glue(self, glue: str):
        """
        Sorts the n-grams by the glue function specified in the glue parameter.
        """
        
        # sort the n-grams by the glue function
        self.n_grams = dict(sorted(self.n_grams.items(), key=lambda item: self.get_glue_value(item[1]), reverse=True))

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
    
    def print_top_n_glue(self, n: int = 10, glue: str = None):
        """
        Prints the top n n-grams by the glue function specified in the glue parameter.
        """
        if glue:
            self.glue_type = glue
        # sort the n-grams by the glue function
        self.sort_by_glue(glue)
        # print the top n n-grams
        for i, (words, data) in enumerate(list(self.n_grams.items())[:n]):
            print(f"{i + 1}: {words} : {data}")

    def print_all(self, glue: str = None):
        """
        Prints the top n n-grams by the glue function specified in the glue parameter.
        """
        if glue:
            self.glue_type = glue
        # sort the n-grams by the glue function
        self.sort_by_glue(glue)
        # print the top n n-grams
        for i, (words, data) in enumerate(list(self.n_grams.items())):
            print(f"{i + 1}: {words} : {data}")

    def evaluate_keywords(pred_keys, true_keys):
        """
            Calculates Precision, Recall, and F1 Score between predicted and true keywords.
        """
        pred_set = set(pred_keys)
        true_set = set(true_keys)
        
        true_positive = len(pred_set & true_set)
        precision = true_positive / len(pred_set) if pred_set else 0
        recall = true_positive / len(true_set) if true_set else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        
        return precision, recall, f1
    
    def print_bottom_n_glue(self, n: int = 10, glue: str = None):
        """
        Prints the bottom n n-grams by the glue function specified in the glue parameter.
        """
        if glue:
            self.glue_type = glue
        # sort the n-grams by the glue function
        self.sort_by_glue(glue)
        # print the bottom n n-grams
        for i, (words, data) in enumerate(list(self.n_grams.items())[-n:]):
            print(f"{i + 1}: {words} : {data}")

    def extract_explicit_keywords(self, top_n: int = 15, glue: str = None):
        """
            Extracts the top-N relevant expressions as explicit keywords.
        """
        if glue:
            self.glue_type = glue
        self.sort_by_glue(glue)
        return list(self.n_grams.keys())[:top_n]
    
    def extract_implicit_keywords(self, explicit_keywords, top_n=10):
        """
            Extracts implicit keywords by computing similarity with explicit ones.
        """
        candidates = [" ".join(words) for words in self.n_grams.keys() if len(words) > 1]
        explicit_texts = [" ".join(words) for words in explicit_keywords]

        vectorizer = TfidfVectorizer().fit(candidates + explicit_texts)
        candidate_vectors = vectorizer.transform(candidates)
        explicit_vectors = vectorizer.transform(explicit_texts)

        sim_matrix = cosine_similarity(candidate_vectors, explicit_vectors)
        scores = sim_matrix.max(axis=1)  # Max similarity to any explicit keyword

        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        return scored_candidates[:top_n]

########################



    def calculate_omega_minus_one(self, ngram):

        n = len(ngram)
        omega_minus = []
        for i in range(n):
            sub_ngram = ngram[:i] + ngram[i+1:]
            if sub_ngram in self.n_grams:
                omega_minus.append(self.get_glue_value(self.n_grams[sub_ngram]))
        return max(omega_minus) if omega_minus else 0.0


    def calculate_omega_plus_one(self, ngram):

        n = len(ngram)
        omega_plus = []
        if n + 1 <= self.n_max:
            for (cand_ngram, cand_data) in self.n_grams.items():
                if len(cand_ngram) == n + 1:
                    for i in range(n + 1):
                        if cand_ngram[i:i + n] == ngram:
                            omega_plus.append(self.get_glue_value(cand_data))
                            break
        return max(omega_plus) if omega_plus else 0.0

    def calculate_neighboring_2grams(self, ngram, data):
        """
        Counts the number of neighboring 2-grams in the corpus for the given unigram and updates the data object.
        """
        n = len(ngram)
        if n != 1:
            return
        count = 0
        for i in range(self.corpus_size - 1):
            if self.corpus[i] == ngram[0]:
                # Check if the previous and next words form a 2-gram 
                if i > 0:
                    prev_word = self.corpus[i - 1]
                    next_word = self.corpus[i + 1] if i + 1 < self.corpus_size else None
                    if next_word is not None:
                        if (prev_word, next_word) in self.n_grams:
                            count += 1
        data.neighboring_2grams = count

    def find_stopwords(self):
        """
        finds stopwords among the unigrams by evaluating the following condition: 
        stopwords have fewer syllables than content words, and have many more neighboring 2-grams 
        than content words. 
        """ 
        for ngram, data in self.n_grams.items():
            if len(ngram) != 1:
                continue
            data.n_syllables = self.calculate_syllables(ngram[0])
            self.calculate_neighboring_2grams(ngram, data)


    

    def calculate_Omegas(self, glue: str = None):
        if glue:
            self.glue_type = glue
        for ngram, data in tqdm(self.n_grams.items(), desc="Calculating Ω values"):
            n = len(ngram)
            if n == 1:
                continue
            data.omega_n_minus_one = self.calculate_omega_minus_one(ngram)
            data.omega_n_plus_one = self.calculate_omega_plus_one(ngram)
        


    def find_MWEs(self, glue: str = None, stopwords=set()):
        if glue:
            self.glue_type = glue
        self.calculate_Omegas(glue)
        p = self.p  # Assume p is defined in your class somewhere
        stopwords = self.stopwords  # Assume this is your stopwords set
        
        MWEs = []

        for ngram, data in self.n_grams.items():
            n = len(ngram)
            if n == 1:
                continue  # skip unigrams

            g_w = self.get_glue_value(data)
            omega_minus = data.omega_n_minus_one
            omega_plus = data.omega_n_plus_one

            # Condition 1 and 2: glue score threshold based on length
            if n == 2:
                cond_glue = g_w >= omega_plus
            else:
                cond_glue = g_w >= ((omega_minus ** p + omega_plus ** p) / 2) ** (1 / p)

            # Condition 3: frequency check
            cond_freq = data.frequency > 1

            # Condition 4: check first and last word not in stopwords
            cond_stopwords = (ngram[0] not in stopwords) and (ngram[-1] not in stopwords)

            if cond_glue and cond_freq and cond_stopwords:
                MWEs.append(ngram)

        return MWEs
