from nltk.corpus import reuters
from time import sleep as wait
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# stopwords = stopwords.words("english")
# print(stopwords)


def is_digit(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def is_math_sign(x):
        if x == "/" or x == "*" or x == "+" or x == "-" or x == "=" or x == "%":
            return True
        else:
            return False


def remove_stop_words(doc_words):
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
                 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
                 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
                 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll',
                 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma',
                 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
    doc_words = [word for word in doc_words if word not in stopwords]

    return doc_words


def process_doc(document_id):
    # Raw document
    raw_doc = reuters.raw(document_id)
    # print(f"Raw document (id:{document_id}): \n" + raw_doc )

    # convert text to lower case
    raw_doc = raw_doc.lower()

    # Replace delimeter signs with whitespaces and split text to a list by whitespaces
    raw_doc_splitted = raw_doc.replace(', ', ' ').replace(',\n', ' ').replace(' "', ' ').replace('" ', ' ').replace('"', ' ')\
                       .replace('. ', ' ').replace('.\n', ' ').replace('(', ' ').replace(')', ' ')\
                       .replace('>', ' ').replace('<', ' ').split()
    # print(raw_doc_splitted) # list of splitted words

    # Replace numbers and math signs with FLAGS
    for word in raw_doc_splitted:
        if word.isdigit():
            raw_doc_splitted[raw_doc_splitted.index(word)] = "__cislo_int__"  # word is int number
        elif is_digit(word):
            raw_doc_splitted[raw_doc_splitted.index(word)] = "__cislo_float__"  # word is float number
        elif is_math_sign(word):
            raw_doc_splitted[raw_doc_splitted.index(word)] = "__mops__"   # word is mathematical operational sign
        else:
            # just word
            pass
    # print(document_id, '\n', raw_doc_splitted)
    return raw_doc_splitted


def create_dictionary(docs, keep_percent=90):
    # Create wordlist from docs
    wordlist = []
    for d in range(len(docs)):
        wordlist.extend(process_doc(docs[d]))

    # Remove stop words
    wordlist_no_sw = remove_stop_words(wordlist)

    # Create freq-dictionary
    wordfreq = [wordlist_no_sw.count(w) for w in wordlist_no_sw]
    dictionary = dict(zip(wordlist_no_sw, wordfreq))
    # print(dictionary)

    # Sort freq-dictionary
    dictionary = [(dictionary[key], key) for key in dictionary]
    dictionary.sort()
    dictionary.reverse()
    # print(dictionary)

    # lenght of the dictionary when we keep ?keep_percent? percent
    percent = (int(len(dictionary) / 100) * keep_percent)
    dictionary = dictionary[0:percent]
    # print(dictionary)

    return dictionary
