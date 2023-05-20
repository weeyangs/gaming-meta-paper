import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import string
from gensim import corpora, models

patch_numbers = ["9.1", "9.2", "9.3", "9.4", "9.5", "9.6", "9.7",\
                "9.8", "9.9", "9.10", "9.11", "9.12", "9.13", "9.14",\
                    "9.15", "9.16", "9.17", "9.18", "9.19", "9.20", "9.21",\
                        "9.22", "9.23", "10.1", "10.2", "10.3", "10.4", "10.5",\
                             "10.6", "10.7", "10.8", "10.9", "10.10", "10.11", "10.12",\
                                 "10.13", "10.14", "10.15", "10.16", "10.17", "10.18", "10.19",\
                                     "11.1", "11.2", "11.3", "11.4", "11.5", "11.6", "11.7", "11.8",\
                                         "11.9", "11.10", "11.11", "11.12", "11.13", "11.14", "11.15",\
                                             "11.16", "11.17", "11.18", "11.19", "11.20", "11.21", "11.22",\
                                                "11.23", "11.24", "12.1", "12.2", "12.3", "12.4", "12.5", "12.6",\
                                                    "12.7", "12.8", "12.9", "12.10", "12.11", "12.12", "12.13", "12.14",\
                                                        "12.15", "12.16", "12.17", "12.18", "12.19", "12.20", "12.21", "12.22",\
                                                            "12.23", "13.1", "13.1b", "13.2", "13.3", "13.4", "13.5", "13.6", "13.7",\
                                                                "13.8", "13.9", "13.10"]


# Define stop words + punctuation + study-specific stop-words
STOP = set(nltk.corpus.stopwords.words('english')
       + list(string.punctuation)
       + patch_numbers
       + ["amp", "39", "subscribe", "follow",
          "link", "game", "league", "legends",
          "music", "champions", "um", "facebook",
          "right", "much", "like", "well", "really",
          "good", "play", "yeah", "also", "would", "still",
          "even", "2023", "know", "want", "sure", "many",
          "actually", "think", "note", "secondary", "video"
         ])

def pos_tag(text):
    '''
    Tags each word in a string with its part-of-speech indicator,
    excluding stop-words and words <= 3 characters

    'I hate this' -> ['i', 'hate', 'this']
    '''
    # Tokenize words using nltk.word_tokenize, keeping only those tokens that do
    # not appear in the stop words we defined
    tokens = [i for i in nltk.word_tokenize(text.lower())
                 if (i not in STOP) and (len(i) > 3)]

    # Label parts of speech automatically using NLTK
    pos_tagged = nltk.pos_tag(tokens)
    return pos_tagged

def plot_top_adj(series, data_description, n = 15):
    '''
    Plots the top `n` adjectives in a Pandas series of strings.
    '''
    # Apply part of Speech tagger that we wrote above to any Pandas series that
    # pass into the function
    pos_tagged = series.apply(pos_tag)

    # Extend list so that it contains all words/parts of speech for all captions
    pos_tagged_full = []
    for i in pos_tagged:
        pos_tagged_full.extend(i)

    # Create Frequency Distribution of diff adjectives and plot distribution
    fd = nltk.FreqDist(word + "/" + tag for (word, tag) in pos_tagged_full
                           if tag[:2] == 'JJ')
    fd.plot(n, title='Top {} Adjectives for '.format(n) + data_description);
    return

def get_wordnet_pos(word):
    '''
    Tags each word with its Part-of-speech indicator -- specifically used for
    lemmatization in the get_lemmas function
    '''
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}

    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

def get_lemmas(text):
    '''
    Gets lemmas for a string input, excluding stop words, punctuation, as well
    as a set of study-specific stop-words

    Only returns lemmas that are greater 3 characters long.
    '''
    lemmas = [nltk.stem.WordNetLemmatizer().lemmatize(t, get_wordnet_pos(t))
              for t in nltk.word_tokenize(text.lower()) if t not in STOP]
    return [l for l in lemmas if len(l) > 3]

def plot_top_lemmas(series, data_description, n = 20):
    '''
    Plots the top `n` lemmas in a Pandas series of strings.
    '''
    lemmas = series.apply(get_lemmas)

    # Extend list so that it contains all words/parts of speech for all captions
    lemmas_full = []
    for i in lemmas:
        lemmas_full.extend(i)

    nltk.FreqDist(lemmas_full).plot(n,
        title='Top {} Lemmas Overall for {}'.format(n, data_description));

def plot_top_tfidf(series, data_description, n=20):
    '''
    Plots the top `n` lemmas (in terms of average TFIDF)
    across a Pandas series (corpus) of strings (documents).
    '''
    # Get lemmas for each row in the input Series
    lemmas = series.apply(get_lemmas)

    # Initialize Series of lemmas as Gensim Dictionary for further processing
    dictionary = corpora.Dictionary(lemmas)

    # Convert dictionary into bag of words format: list of
    # (token_id, token_count) tuples
    bow_corpus = [dictionary.doc2bow(text) for text in lemmas]

    # Calculate TFIDF for each word in a document,
    # and compute total TFIDF sum across all documents:
    tfidf = models.TfidfModel(bow_corpus, normalize=True)
    tfidf_weights = {}
    for doc in tfidf[bow_corpus]:
        for ID, freq in doc:
            tfidf_weights[dictionary[ID]] = np.around(freq, decimals=2) \
                                          + tfidf_weights.get(dictionary[ID], 0)

    # highest (average) TF-IDF:
    top_n = pd.Series(tfidf_weights).nlargest(n) / len(lemmas)

    # Plot the top n weighted words:
    plt.plot(top_n.index, top_n.values)
    plt.xticks(rotation='vertical')
    plt.title('Top {} Lemmas (TFIDF) for {}'.format(n, data_description));
