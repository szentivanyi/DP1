from nltk.corpus import reuters
import nltk
# nltk.download('reuters')


def create_lvs(docs):
    all_topics = reuters.categories()
    documents_ids = reuters.fileids()[:docs]
    train_docs = []
    test_docs = []
    lvs = []

    # for id in documents_ids:
        # if id.startswith("train"):
        #     train_docs.append(reuters.raw(id))
        # else:
        #     test_docs.append(reuters.raw(id))

    for id in documents_ids:

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