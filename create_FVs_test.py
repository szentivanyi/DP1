from nltk.corpus import reuters
# import nltk
# from nltk.corpus import stopwords
# import re

# http://disi.unitn.it/moschitti/corpora.htm  ine CORPUSy
# http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
# https://github.com/niderhoff/nlp-datasets   NLP DATASETY

# nltk.download('stopwords')
# stopwords = stopwords.words("english")
# print(stopwords)

train_docs = []
test_docs = []
# rozdelim si databazu dokumentov do trenovacej a testovacej mnoziny (list-u)
for doc_id in reuters.fileids():
    if doc_id.startswith("train"):
        train_docs.append(doc_id)
    else:
        test_docs.append(doc_id)


# documents = reuters.fileids()  # 21 578 docs v 90 kategoriach

########################################
###### 1. STEP: CREATE DICTIONARY ######
########################################

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
    documents = test_docs[:docs]
    # Create wordlist from docs
    wordlist = []
    for d in range(docs):
        wordlist.extend(process_doc(documents[d]))
    # print(wordlist)

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


################################################
#### 1. Create Feature-Vector for each doc #####
################################################

def create_fvs_test(docs):
    # vytvor slovnik z poctu 'docs' dokumentov a zachovaj len 80% slovnika
    dictionary = create_dictionary(docs, keep_percent=80)

    documents = test_docs[:docs]

    fvs = []
    for doc in range(docs):
        # Process 1 raw document to list without numbers, signs, etc.
        wordlist = process_doc(documents[doc])
        len_wordlist = len(wordlist)

        # Remove stop words
        wordlist = remove_stop_words(wordlist)
        len_wordlist_no_sw = len(wordlist)

        # number of stop words in document
        sw_count = len_wordlist - len_wordlist_no_sw

        # Replace words from doc not included in our dictionary by __OOV__
        fv = []
        fv_nn = []
        for word in dictionary:   # v tvare list slovnikov [ {2:x} , {5:y} ...]
            if word[-1] in wordlist:  # x, y
                count = wordlist.count(word[-1])  # pocet vyskytov daneho slova v dokumente
                fv.append(dict({count:word[-1]}))  # human readable FV
                fv_nn.append(count)     # NN readable FV
            else:
                fv.append('__OOV__')
                fv_nn.append(0)

        # pocet slov dokumentu pripojime na koniec FV
        fv.append(len_wordlist)
        fv_nn.append(len_wordlist)

        # pocet stop slov na dokument v percentach pripojime na koniec FV
        fv.append(float("{0:.2f}".format((sw_count/len_wordlist)*100)))
        fv_nn.append(float("{0:.2f}".format((sw_count/len_wordlist)*100)))

        fvs.append(fv_nn)

        # print("Feature vector: ")
        # print('dlzka: ' + str(len(fv)))
        # print(fv)

    return fvs  # all Feature Vectors as list




