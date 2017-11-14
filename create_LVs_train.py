from nltk.corpus import reuters


def create_lvs_train(train_docs):
    all_topics = reuters.categories()

    lvs = []
    for id in train_docs:
        doc_topic = reuters.categories(id)
        lv = []
        for topic in all_topics:
            if topic in doc_topic:
                lv.append(1)
            else:
                lv.append(0)

        # print(len(lv))
        # print(lv)

        lvs.append(lv)

    # print(len(lvs))
    # print(lvs)

    return lvs  # all Label Vectors as list