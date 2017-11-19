COUNT_PROCESS_DOCS = 1000     # Number of documents to process as Feature vectors/Label vectors
COUNT_DICTIONARY_DOCS = 500  # Number of documents to create dictionary


import csv
from nltk.corpus import reuters
from create_train_data import create_train_data  # Import train data
from create_test_data import create_test_data  # Import test data
from create_dictionary import create_dictionary


# rozdelim si db dokumentov do trenovacej a testovacej mnoziny (list-u)
categories = reuters.categories()
train_docs = []
test_docs = []
for doc_id in reuters.fileids():
    doc_topics = reuters.categories(doc_id)  # najdem topiky dokumentu
    if len(doc_topics) == 1:  # ak ma dokument len 1 topic  (niektore maju 0 alebo 2-3)
        if doc_id.startswith("train"):
            train_docs.append(doc_id)
        else:
            test_docs.append(doc_id)

print("Topikov/kategorii: " + str(len(categories)))
print("Training docs: " + str(len(train_docs)))
print("Testing docs: " + str(len(test_docs)))


# vytvorim slovnik
dictionary = create_dictionary(train_docs[:COUNT_DICTIONARY_DOCS], keep_percent=80)

# vytvorim train a test data vektory
train_data = create_train_data(train_docs[:COUNT_PROCESS_DOCS], dictionary)
test_data = create_test_data(train_docs[:COUNT_PROCESS_DOCS], dictionary)
# print(train_data)
# print(test_data)
# print("----------")


# vytvorim train_data.CSV
with open('train_data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow([len(train_data), len(train_data[0])-1, categories])  # HLAVICKA CSV: pocet riadkov/dokumentov, pocet vlastnosti = dlzka 1 FV, vypisane vsetky kategorie
    writer.writerows(train_data)    # Feature data + posledny stlpec v riadku je Label

# vytvorim test_data.CSV
with open('test_data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow([len(test_data), len(test_data[0])-1, categories]) # HLAVICKA CSV
    writer.writerows(test_data)  # Feature data + posledny stlpec v riadku je Label

print("OK, documents was created!")