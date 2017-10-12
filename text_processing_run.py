from nltk import word_tokenize
from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

# musim si najprv stiahnut stopwords, reuters, punkt hlada ich na disku a nenajde
import nltk
nltk.download('stopwords')
nltk.download('reuters')
nltk.download('punkt')

cachedStopWords = stopwords.words("english")  # return lemmas of the given language as list of words zo zoznamu stop slov
print("Nakešované stop slová: ")
print(cachedStopWords)


def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))  # prevediem do lowercase
    words = [word for word in words if word not in cachedStopWords]  # vynecham stop slova
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))  # odstranim sklonovanie = zakl. tvar
    p = re.compile('[a-zA-Z]+')      # vytvorim regex - len pismena
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens)) # vyfiltrujem len slova zlozene z pismen a dlhsie ako 3 znaky
    #print("filtered_tokens: ")
    #print(filtered_tokens)

    return filtered_tokens


# Return the representer, without transforming
# Convert a collection of raw documents to a matrix of TF-IDF features
def tf_idf(docs):
    tfidf = TfidfVectorizer(tokenizer=tokenize, min_df=3, max_df=0.90, max_features=1000, use_idf=True,
                            sublinear_tf=True)
    tfidf.fit(docs)  # Learn vocabulary and idf from training set

    # print(type(tfidf))
    # print("tfidf: ")
    # print(tfidf)
    return tfidf


def feature_values(doc, representer):  # vstup: testovaci dokument a feature vector ziskany z trenovacich dokumentov
    doc_representation = representer.transform([doc])  # return Sparse matrix, [n_samples, n_features],   Tf-idf-weighted document-term matrix.
    features = representer.get_feature_names()  # Returns a list of feature names, ordered by their indices

    # len pekny vypis meno vlastnosti a dokument
    return [(features[index], doc_representation[0, index]) for index in doc_representation.nonzero()[1]]


def collection_stats():
    # List of documents
    documents = reuters.fileids()
    print(str(len(documents)) + " documents")

    train_docs = list(filter(lambda doc: doc.startswith("train"), documents))
    print(str(len(train_docs)) + " total train documents")

    test_docs = list(filter(lambda doc: doc.startswith("test"), documents))
    print(str(len(test_docs)) + " total test documents")

    # List of categories
    categories = reuters.categories()
    print(str(len(categories)) + " categories")

    # Documents in a category
    category_docs = reuters.fileids("acq")

    # Words for a document
    document_id = category_docs[0]
    document_words = reuters.words(category_docs[0])  # words: the given file(s) as a list of words and punctuation symbols.

    print("document_words: ")
    print(document_words)

    # Raw document
    print("reuters.raw(document_id) :")
    print(reuters.raw(document_id))


def main():
    train_docs = []
    test_docs = []
    print('STATISTIKY: ')
    print(collection_stats())

    # rozdelim si databazu dokumentov do trenovacej a testovacej mnoziny (list-u)
    for doc_id in reuters.fileids():
        if doc_id.startswith("train"):
            train_docs.append(reuters.raw(doc_id))
        else:
            test_docs.append(reuters.raw(doc_id))

    # z trenovacich dokumentov si vytvorime feature vector
    representer = tf_idf(train_docs)

    # z test dokumentov si vytvorime feature vector
    for doc in test_docs:
        #print(feature_values(doc, representer))
        pass

# aby spustenie text_processing_run.py zavolalo metodu main() a nechapalo subor ako modul na import
if __name__ == "__main__":
    main()