from NGramData import NGramData
from tqdm import tqdm
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from psutil import cpu_count # for os.cpu_count()

# outside of class so that it's picklable -> we're able to pass it to another process to parallelize
def get_glue_value(glue_type, data: NGramData):
    return {
        "scp": data.scp,
        "dice": data.dice,
        "phi_square": data.phi_square
    }[glue_type]

# function to extract relevant n-grams for omega calculations
# needed to define only the relevant n-grams for each n-gram in the corpus, 
# to avoid sending the full n-gram dictionary to each process and have mamory issues
def extract_relevant_ngrams(chunk, full_n_grams_by_len, n_max):
    relevant_keys = set(chunk)
    for ngram in chunk:
        n = len(ngram)
        if n > 1:
            for i in range(n):
                sub_ngram = ngram[:i] + ngram[i + 1:]
                relevant_keys.add(sub_ngram)
        if n + 1 <= n_max:
            for cand_ngram in full_n_grams_by_len.get(n + 1, []):
                if len(cand_ngram) == n + 1:
                    for i in range(n + 1):
                        if cand_ngram[i:i + n] == ngram:
                            relevant_keys.add(cand_ngram)
                            break
    # Trim to only relevant n_grams
    return {k: full_n_grams_by_len[k] for k in relevant_keys if k in full_n_grams_by_len}



# parallel function to compute omega values for n-grams
def compute_omega_chunk(ngram_chunk, n_grams, glue_type, n_max):
    results = []
    for ngram in ngram_chunk:
        n = len(ngram)

        omega_minus = []
        for i in range(n):
            sub_ngram = ngram[:i] + ngram[i + 1:]
            if sub_ngram in n_grams:
                omega_minus.append(get_glue_value(glue_type, n_grams[sub_ngram]))
        omega_n_minus_one = max(omega_minus) if omega_minus else 0.0

        omega_n_plus_one = 0.0
        if n + 1 <= n_max:
            for cand_ngram, cand_data in n_grams.items():
                if len(cand_ngram) == n + 1:
                    for i in range(n + 1):
                        if cand_ngram[i:i + n] == ngram:
                            omega_n_plus_one = max(
                                omega_n_plus_one, get_glue_value(glue_type, cand_data)
                            )
                            break
        results.append((ngram, omega_n_minus_one, omega_n_plus_one))
    return results

class Extractor:
    """
    The Extractor class is responsible for extracting n-grams from a given corpus.
    Stores n-grams in a defaultdict collection, as it's more concise to count elements with it (easier to add 
    non-existing keys). The dictionary is formed by tuples of tokens as keys (the words forming the n-gram), 
    and the corresponding n-gram object as value. In this way we can access frequency and all other metadata of
    the n-gram in O(1) by dictionary look-up on the words tuple.
    """

    def __init__(self, corpus, n_max: int = 7, limit: int = None, glue_type: str = "scp", p: int = 2):
        
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.n_max = n_max
        self.n_grams = defaultdict(NGramData)
        self.find_n_grams(limit) 
        self.total_ngrams = defaultdict(int) # to count the total number of n-grams of each size
        self.find_total_ngrams()
        self.glue_functions = {
            "scp": self.calculate_scp,
            "dice": self.calculate_dice,
            "phi_square": self.calculate_phi_square
        }
        self.glue_type = glue_type  # type of glue function to use for sorting and filtering n-grams
        self.p = p # parameter for glue functions that require it, e.g., SCP
        self.MWEs = []  # to store multi-word expressions (MWEs) found in the corpus
        self.n_grams_by_len = defaultdict(list)
        self.compute_n_grams_by_len()  # to store n-grams by their length for quick access

        
    def compute_n_grams_by_len(self):
        """
        Computes and stores n-grams by their length in the n_grams_by_len attribute.
        This allows for quick access to n-grams of a specific length.
        """
        for ngram in self.n_grams:
            self.n_grams_by_len[len(ngram)].append(ngram)

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
        Computes and assigns glue values for all n-grams (of length > 1) using the defined glue function.
        Each result is saved in the corresponding attribute of the NGramData object.
        """
        glue_func = self.glue_functions.get(self.glue_type)
        for words, data in tqdm(self.n_grams.items(), desc="Calculating glue values", unit="n-gram", mininterval=50000):
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
    

    def sort_by_glue(self):
        """
        Sorts the n-grams by the glue function specified in the glue parameter.
        """
        
        # sort the n-grams by the glue function
        self.n_grams = dict(sorted(self.n_grams.items(), key=lambda item: get_glue_value(self.glue_type,item[1]), reverse=True))

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
        Extracts the top-N relevant expressions as explicit keywords from MWEs.
        """
        if glue:
            self.glue_type = glue
        self.sort_by_glue(glue)
        return list(self.MWEs)[:top_n]

    def extract_implicit_keywords(self, explicit_keywords, top_n=10):
        """
        Extracts implicit keywords by computing similarity with explicit ones on MWEs.
        """
        # Candidates are MWEs with length > 1
        candidates = [" ".join(mwe) for mwe in self.MWEs if len(mwe) > 1]
        explicit_texts = [" ".join(mwe) for mwe in explicit_keywords]

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
                omega_minus.append(get_glue_value(self.glue_type, self.n_grams[sub_ngram]))
        return max(omega_minus) if omega_minus else 0.0


    def calculate_omega_plus_one(self, ngram):

        n = len(ngram)
        omega_plus = []
        if n + 1 <= self.n_max:
            for (cand_ngram, cand_data) in self.n_grams.items():
                if len(cand_ngram) == n + 1:
                    for i in range(n + 1):
                        if cand_ngram[i:i + n] == ngram:
                            omega_plus.append(get_glue_value(self.glue_type, cand_data))
                            break
        return max(omega_plus) if omega_plus else 0.0


    def calculate_Omegas(self):

        for ngram, data in tqdm(self.n_grams.items(), desc="Calculating Ω values",
                                 unit="n-gram", miniters=200):
            n = len(ngram)
            if n == 1:
                continue
            data.omega_n_minus_one = self.calculate_omega_minus_one(ngram)
            data.omega_n_plus_one = self.calculate_omega_plus_one(ngram)
        
    
    def parallel_calculate_Omegas(self, num_workers: int = None):

        # Split n-gram keys into chunks
        ngram_list = list(self.n_grams.keys())
        chunk_size = math.ceil(len(ngram_list) / num_workers)
        ngram_chunks = [ngram_list[i:i + chunk_size] for i in range(0, len(ngram_list), chunk_size)]

        # Prepare only the necessary subset of n_grams for each chunk
        chunk_args = []
        for chunk in tqdm(ngram_chunks, desc="Preparing chunks for parallel processing", unit="chunk"):
            relevant_subset = extract_relevant_ngrams(chunk, self.n_grams, self.n_max)
            chunk_args.append((chunk, relevant_subset, self.glue_type, self.n_max))

        # Run in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(compute_omega_chunk, *args)
                for args in chunk_args
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating Ω", unit="process"):
                for ngram, omega_minus, omega_plus in future.result():
                    if ngram in self.n_grams and omega_minus is not None:
                        self.n_grams[ngram].omega_n_minus_one = omega_minus
                        self.n_grams[ngram].omega_n_plus_one = omega_plus

    def find_MWEs(self, stopwords=set(), parallel: bool = False, n_workers: int = None):
        
        glue = self.glue_type  
        if parallel:
            if n_workers is None:
                n_workers = max(1, cpu_count(logical=False) - 1)
            tqdm.write(f"Using {n_workers} workers for parallel processing.")
            self.parallel_calculate_Omegas(n_workers)
        else:
            self.calculate_Omegas(glue)
        p = self.p  # Assume p is defined in your class somewhere
        
        MWEs = []

        for ngram, data in tqdm(self.n_grams.items(), desc="Finding MWEs", unit="n-gram"):
            n = len(ngram)
            if n == 1:
                continue  # skip unigrams

            g_w = get_glue_value(self.glue_type, data)
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

        self.MWEs = MWEs

