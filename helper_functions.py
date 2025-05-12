from collections import Counter
from tqdm import tqdm
import os
from enum import Enum, auto




def load_and_preprocess_corpus(dir_path: str, special_chars: "list[str]" = None):
    """
    Opens the corpus from a directory, and preprocesses it by preappending space on left of special chars
    Args:
        dir_path (str): The path to the directory containing the text files.
        special_chars (list[str]): A list of special characters to preprocess.
    Returns:
        corpus (str): The preprocessed text corpus.
    """
    """
    TODO, ISSUES:
    - cuts numbers like ",000"
    - cuts urls
    - leaves consecutive special chars
    """

    corpus = ""
    for file in tqdm(os.listdir(dir_path), unit="files", desc="Loading corpus"):
        with open(os.path.join(dir_path, file), "r", encoding="utf-8") as f:
            document = f.read()
            # Preprocess the corpus by adding space before special characters
            for char in special_chars:
                try: # as document.index() throws exceptions if char is not found
                    # Check if the character is in the document and add space before it.
                    # Dots are replaced only if the preceding substring has no capital letters, 
                    # to avoid breaking acronyms or words like "Dr.".
                    # space is added after new line characters if the next character is not a space as newlines are 
                    # found at the beginning of files
                    document = document.replace(char,
                                                " " + char if document[document.index(char) - 1] != " "
                                                    and not (char == '.' and is_previous_substring_capitalized(document, document.index(char) - 1))  
                                                # else char + " " if char == "\n" and document[document.index(char) + 1] != " "
                                                else char + " " if document[document.index(char) + 1] != " " 
                                                else char)
                    document = document.replace("\n", " \n ") # add space before and after new line characters

                except ValueError as ve:
                    pass
                except IndexError as ie:
                   pass
            corpus += document
    corpus = corpus.split(" ")
    return [word for word in corpus if word != "" and word != "\n"]
     

def is_previous_substring_capitalized(corpus: str, index: int) -> bool:
    """
    Checks if the substring before the index is capitalized.
    
    Args:
        corpus (str): The text corpus to analyze.
        index (int): The index to check.
    Returns:
        bool: True if the substring before the index is capitalized, False otherwise.
    """
    # Check if the substring before the index is uppercase
    is_upper = False
    while index > 0 and corpus[index] != " ":
        if corpus[index].isupper():
            is_upper = True
        index -= 1
    return is_upper

def find_words_frequencies(corpus: str):
    """
    Finds the frequency of each word in the corpus, showing progress bar.
    
    Args:
        corpus (str): The text corpus to analyze.
    Returns:
        dict: A dictionary with words as keys and their frequencies as values.
    """

    word_counter = {}
    # Tokenize each sentence (based on whitespace) and update the frequency count
    for word in corpus:
        # ignore newline characters
        if word in word_counter.keys():
            word_counter[word] += 1
        else:
            word_counter[word] = 1

    # Sort the dictionary by frequency in descending order
    word_counter = dict(sorted(word_counter.items(), key=lambda item: item[1], reverse=True))
    return word_counter


