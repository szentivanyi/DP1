from nltk.corpus import reuters
import nltk


# nltk.download('reuters')


def create_lvs_train(docs):
    all_topics = reuters.categories()

    # rozdelim si databazu dokumentov do trenovacej a testovacej mnoziny (list-u)
    train_docs = []
    test_docs = []
    for doc_id in reuters.fileids():
        if doc_id.startswith("train"):
            train_docs.append(doc_id)
        else:
            test_docs.append(doc_id)

    documents_ids = train_docs[:docs]

    lvs = []
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