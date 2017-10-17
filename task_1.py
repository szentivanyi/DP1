from nltk.corpus import reuters
import nltk
from nltk.corpus import stopwords
import re

documents = reuters.fileids()
# nltk.download('stopwords')
# stopwords = stopwords.words("english")
# print(stopwords)

###################333333333#############
###### 1. STEP: CREATE DICTIONARY ######
##################33333333###############


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

    # Replace delimeter signs with whitespaces and split text to list by whitespaces
    raw_doc_splitted = raw_doc.replace(',', ' ').replace(' "', ' ').replace('" ', ' ').replace('"', ' ')\
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


def create_dictionary(docs, cut_percent=90):
    # Create wordlist from all docs
    wordlist = []
    for i in range(docs):
        wordlist.extend(process_doc(documents[i]))
    # print(wordlist)

    # Remove stop words
    wordlist = remove_stop_words(wordlist)
    # print(doc_wordlist)

    # Create freq-dictionary
    wordfreq = [wordlist.count(p) for p in wordlist]
    dictionary = dict(zip(wordlist, wordfreq))

    # Sort freq-dictionary
    dictionary = [(dictionary[key], key) for key in dictionary]
    dictionary.sort()
    dictionary.reverse()

    # Shorten to XX percent of the dictionary length
    percent = (int(len(dictionary) / 100) * cut_percent)
    dictionary = dictionary[0:percent]
    # print(dictionary)

    return dictionary


################################################
#### 2. Create Feature-Vector for each doc #####
################################################

def create_feature_vector(doc, dictionary):
    # Process 1 raw document to list without numbers, signs, etc.
    wordlist = process_doc(documents[doc])
    # print(wordlist)

    # Remove stop words
    wordlist = remove_stop_words(wordlist)
    # print(wordlist)

    # Replace words not included in vocabulary by __OOV__
    # print("Words not in dictionary: ")
    for word in wordlist:
        s = ''
        # prejdem cely slovnik a lepim znaky F (false) do stringu, ak nenachadzam v slovniku hladane slovo,
        # ak je cely string zlozeny len z F, mal by potvrdit ze slovo nie je v slovniku
        for x in dictionary:
            if word in x[1]:
                s += 'T'
            else:
                s += 'F'
        if 'T' not in s:  # slovo nie je v slovniku
            # print(word)
            wordlist[wordlist.index(word)] = "*__OOV__*"  # word is Out of Vocabulary
    # print("New wordlist: ")
    # print(wordlist)

    # Create freq-dictionary
    wordfreq = [wordlist.count(p) for p in wordlist]
    dictionary = dict(zip(wordlist, wordfreq))

    # Sort freq-dictionary
    dictionary = [(dictionary[key], key) for key in dictionary]
    dictionary.sort()
    dictionary.reverse()

    # print(f"Feature vector of document {i}: ")
    # print(dictionary)

    return dictionary  # as Feature Vector


################################################
#### 3. Run      #####
################################################
docs = 100
fvs = []
dictionary = create_dictionary(docs=docs, cut_percent=80)
print("Dictionary length: ")
print(len(dictionary))
print(dictionary)

print("Feature vectors: ")
for i in range(docs):
    fv = create_feature_vector(doc=i, dictionary=dictionary)
    # print(len(fv))
    # print(fv)
    if len(fv) < len(dictionary):
        fv += [None] * (len(dictionary) - len(fv))
        # print(len(fv))
        # print(fv)
        # print()
    fvs.append(fv)


print("ALL FVs: ")
print(len(fvs))
print(fvs)


