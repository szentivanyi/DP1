from create_dictionary import process_doc, remove_stop_words
from nltk.corpus import reuters


def create_train_data(docs, dictionary):
    all_topics = reuters.categories()
    train_data = []
    for i in range(len(docs)):
        # Process 1 raw document to list without numbers, signs, etc.
        wordlist = process_doc(docs[i])
        len_wordlist = len(wordlist)

        # Remove stop words
        wordlist = remove_stop_words(wordlist)
        len_wordlist_no_sw = len(wordlist)

        # Number of stop words in document
        sw_count = len_wordlist - len_wordlist_no_sw

        # Replace words from doc not included in our dictionary by __OOV__
        fv = []
        fv_nn = []
        for word in dictionary:   # v tvare list slovnikov [ {2:x} , {5:y} ...]
            if word[-1] in wordlist:  # x, y
                count = wordlist.count(word[-1])  # pocet vyskytov daneho slova v dokumente
                # fv.append(dict({count:word[-1]}))  # human readable FV
                # fv_nn.append(count)     # NN readable FV
                fv_nn.append(((count/len_wordlist)*100))     # vyskyt daneho slova v dokumente v %
            else:
                # fv.append('__OOV__')
                fv_nn.append(0.)

        # Pocet vsetkych slov dokumentu, pripojime na koniec FV
        # fv.append(len_wordlist)
        fv_nn.append(len_wordlist)

        # Pocet stop-slov v dokumente, v percentach pripojime na koniec FV
        # fv.append(float("{0:.2f}".format((sw_count/len_wordlist)*100)))
        # fv_nn.append(float("{0:.2f}".format((sw_count/len_wordlist)*100)))
        fv_nn.append(((sw_count/len_wordlist)*100))

        # Na akej pozicii sa nachadza topic v zozname vsetkych topicov, jeho pozicia je UNIQUE, pripojime na koniec FV
        doc_topic = reuters.categories(docs[i])
        topic_id = all_topics.index(doc_topic[0])
        fv_nn.append(topic_id)

        # V kazdom cykle spracujeme 1 dokument a prilepime jeho vektor vlastnosti + topic id k trenovacim datam
        train_data.append(fv_nn)

    return train_data  # all Feature Vectors as list + result topic





