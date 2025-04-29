from collections import Counter
from tqdm import tqdm
import os

def find_frequencies(dir_path: str, special_chars: "list[str]" = None):
    """
    Opens the corpus from a directory, and finds the frequency of each word in the corpus, showing progress bar.
    
    Args:
        dir_path (str): The path to the directory containing text files.
        
    Returns:
        dict: A dictionary with words as keys and their frequencies as values.
    """
    word_counter = {}
    
    # Opens all files in the directory and adds the frequency of each word to the dictionary
    for filename in tqdm(os.listdir(dir_path), desc="Counting words in corpus", unit="file"):

        with open(os.path.join(dir_path, filename), "r", encoding="latin-1") as file:
            corpus = file.readlines()
        
        # Tokenize each sentence (based on whitespace) and update the frequency count
        for sentence in corpus:
            words = sentence.split(" ")
            for word in words:
                # ignore newline characters
                if word == "\n":
                    continue

            # preappend space on left of special chars :
                for special_char in special_chars:
                    if special_char in word:

                        # dots are special chars only if they are not at the end of the word
                        if special_char == ".":
                            if "." in word and word[-1] != ".":
                                # surround dot with spaces
                                word = word.replace(".", " .")
                        else:
                            word = word.replace(special_char, " " + special_char)

                if word in word_counter.keys():
                    word_counter[word] += 1
                else:
                    word_counter[word] = 1

    # Sort the dictionary by frequency in descending order
    word_counter = dict(sorted(word_counter.items(), key=lambda item: item[1], reverse=True))
    return word_counter


