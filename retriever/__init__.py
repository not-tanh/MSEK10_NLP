import pickle
import time

from rank_bm25 import BM25Okapi

from config import STOPWORDS_PATH


class Retriever:
    def __init__(self, data_path, model_path='bm25_model.pkl'):
        with open(data_path, 'r', encoding='utf-8') as file:
            self.library = file.read().split("=====================================\n")

        # check if you already have the previous BM25 model in the file and save bm25 model
        try:
            with open(model_path, 'rb') as model_file:
                self.bm25 = pickle.load(model_file)
        except FileNotFoundError:
            from preprocessing import Preprocessor
            print('Preprocessing documents...')
            preprocessor = Preprocessor(STOPWORDS_PATH)
            count = 0
            with open('cleaned_data.txt', 'w') as f:
                for document in self.library:
                    count += 1
                    if count % 1000 == 0:
                        print(count)
                    f.write(preprocessor.clean_text(document).lower())
                    f.write('=====================================\n')

            with open('cleaned_data.txt', 'r') as f:
                tokenized_library = f.read().split('=====================================\n')

            print('Indexing documents...')
            self.bm25 = BM25Okapi([doc.split() for doc in tokenized_library])
            with open(model_path, 'wb') as model_file:
                pickle.dump(self.bm25, model_file)
            print('Done.')

    def get_document_by_id(self, doc_id) -> str:
        return self.library[doc_id]

    def find_relevant_documents(self, query, k=5) -> list:
        t = time.time()
        query = query.lower()
        print(query)
        scores = self.bm25.get_scores(query.split())
        print(scores)
        ranked_documents = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_documents = ranked_documents[:k]

        result = []
        for doc_id, score in top_documents:
            tmp = {'doc_id': doc_id, 'score': score, 'context': self.get_document_by_id(doc_id)}
            result.append(tmp)
        print(f'Query time: {time.time() - t}')
        return result

# Split title and content
# titles = []
# contents = []
# for section in library:
#     first_newline = section.find('\n')
#     if first_newline != -1:
#         title = section[:first_newline]
#         content = section[first_newline:].strip()
#     else:
#         title = section.strip()
#         content = ""
#     titles.append(title)
#     contents.append(content)
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
