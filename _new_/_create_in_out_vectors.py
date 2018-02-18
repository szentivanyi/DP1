# vytvareme vektory, kde posledny prvok hovori o tom, do akej kategorie text patri, zvysne prvky urcuju vlastnosti dokumentu
from create_dictionary import process_doc, remove_stop_words
from nltk.corpus import reuters


def create_in_out_vectors(docs, dictionary):
    all_topics = reuters.categories()


    #######################################
    #### Input vector / Feature Vector ####
    #######################################
    vectors = []
    for i in range(len(docs)):
        # Process 1 raw document to wordlist without numbers, signs, etc.
        wordlist = process_doc(docs[i])
        len_wordlist = len(wordlist)

        # Remove stop words from wordlist = no sw
        wordlist_no_sw = remove_stop_words(wordlist)
        len_wordlist_no_sw = len(wordlist_no_sw)

        # number of stop words in document
        sw_count = len_wordlist - len_wordlist_no_sw

        # Replace words in wordlist (document), which are not included in our dictionary by 0 (or __OOV__ in human readable format)
        fv = []  # human readable
        fv_nn = []  # nn readable
        for word in dictionary:   # v tvare list slovnikov [{12x : car}, {5x : bank}...]
            if word[-1] in wordlist_no_sw:  # slovoX
                # pocet vyskytov daneho slova vo wordliste (dokumente)
                count = wordlist_no_sw.count(word[-1])
                # todo zjednodus if count nie je 0
                # fv.append(dict({count:word[-1]}))  # human readable FV
                # vyskyt daneho slova vo wordliste v percentach
                fv_nn.append(((count/len(wordlist_no_sw))*100))
            else:
                # fv.append('__OOV__')
                fv_nn.append(0.) # slovo zo slovnika v texte nie je

        # Pocet vsetkych slov nespracovaneho dokumentu, pripojime na koniec FV
        # fv.append(len_wordlist)
        fv_nn.append(len_wordlist)

        # Pocet stop-slov v (nespracovanom) dokumente, pripojime na koniec FV v percentach
        # fv.append(float("{0:.2f}".format((sw_count/len_wordlist)*100)))
        # fv_nn.append(float("{0:.2f}".format((sw_count/len_wordlist)*100)))
        fv_nn.append((sw_count/len_wordlist)*100)

        #########################
        #### Output / Label  ####
        #########################
        # na koniec FV pripojime LABEL=TOPIC dokumentu (na akej pozicii sa nachadza topic v zozname vsetkych topicov, je UNIQUE, pouzijeme ako topic_id)
        doc_topic = reuters.categories(docs[i]) # aky ma topic dokument i
        topic_id = all_topics.index(doc_topic[0]) # ake je ID daneho topicu
        fv_nn.append(topic_id)  # id topicu priplepime na koniec FV


        ####################################
        ### Vectors in list for documents###
        ####################################
        # V kazdom cykle spracujeme 1 dokument a pridame jeho vector (FV+LV) do listu vsetkych vectorov
        vectors.append(fv_nn)

    return vectors  # all vectors (Feature Vectors + Labels) as list