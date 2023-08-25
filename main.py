import streamlit as st

from config import MODEL_PATH, DATA_PATH, STOPWORDS_PATH
from preprocessing import Preprocessor
from retriever import Retriever
from reader.predictor import Predictor, QuestionContextInput


@st.cache_resource
def load_resources():
    p = Preprocessor(STOPWORDS_PATH)
    r = Retriever(DATA_PATH)
    m = Predictor(MODEL_PATH)
    return p, r, m


with st.spinner('Đang tải mô hình...'):
    preprocessor, retriever, predictor = load_resources()


with st.sidebar:
    st.title('💬 Hỏi đáp wiki')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hỏi tôi đi"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hỏi tôi đi"}]


st.sidebar.button('Xóa lịch sử chat', on_click=clear_chat_history)

user_msg = st.chat_input('Nhập câu hỏi')
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message('user'):
        st.write(user_msg)

    documents = retriever.find_relevant_documents(preprocessor.clean_text(user_msg), k=5)

    st.sidebar.write('Các tài liệu liên quan:')
    st.sidebar.write(documents)
    qci = QuestionContextInput(question=user_msg, context=documents[0]['context'])
    answer = predictor.answer([qci])
    print(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer['answer']})
    with st.chat_message('assistant'):
        st.write(answer['answer'])
