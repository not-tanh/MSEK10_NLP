import pickle
import time

from rank_bm25 import BM25Okapi

from config import DATA_PATH

with open(DATA_PATH, 'r', encoding='utf-8') as file:
    library = file.read().split("=====================================\n")
library = [section.lower() for section in library]

# Split title and content
titles = []
contents = []
for section in library:
    first_newline = section.find('\n')
    if first_newline != -1:
        title = section[:first_newline]
        content = section[first_newline:].strip()
    else:
        title = section.strip()
        content = ""
    titles.append(title)
    contents.append(content)

# check if you already have the previous BM25 model in the file and save bm25 model
try:
    with open('bm25_model.pkl', 'rb') as model_file:
        bm25 = pickle.load(model_file)
except FileNotFoundError:
    tokenized_library = [document.split() for document in library]
    bm25 = BM25Okapi(tokenized_library)
    with open('bm25_model.pkl', 'wb') as model_file:
        pickle.dump(bm25, model_file)


def get_document_by_id(doc_id) -> dict:
    return {'title': titles[doc_id], 'content': contents[doc_id]}


def find_relevant_documents(query, k=5) -> list:
    t = time.time()
    query = query.lower()
    scores = bm25.get_scores(query.split())
    ranked_documents = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_documents = ranked_documents[:k]

    result = []
    for doc_id, score in top_documents:
        tmp = {'doc_id': doc_id, 'score': score}
        tmp.update(get_document_by_id(doc_id))
        result.append(tmp)
    print(f'Query time: {time.time() - t}')
    return result


#
# if __name__ == '__main__':
#     query = "quảng trường duyệt binh"
#
#     scores = bm25.get_scores(query.split())
#     ranked_documents = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
#
#     ranking_file = 'ranking.txt'
#     with open(ranking_file, 'w', encoding='utf-8') as file:
#         for doc_id, score in ranked_documents:
#             file.write(f"Document {doc_id}: Score {score}\n")
#
#     print("Ranking file has been created: ranking.txt")
