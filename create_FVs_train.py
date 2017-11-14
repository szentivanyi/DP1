from create_dictionary import process_doc, remove_stop_words


def create_fvs_train(docs, dictionary):
    train_fvs = []
    for i in range(len(docs)):
        # Process 1 raw document to list without numbers, signs, etc.
        wordlist = process_doc(docs[i])
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

        # pocet stop-slov v dokumente, v percentach pripojime na koniec FV
        fv.append(float("{0:.2f}".format((sw_count/len_wordlist)*100)))
        fv_nn.append(float("{0:.2f}".format((sw_count/len_wordlist)*100)))

        train_fvs.append(fv_nn)

        # print("Feature vector: ")
        # print('dlzka: ' + str(len(fv)))
        # print(fv)

    return train_fvs  # all Feature Vectors as list




