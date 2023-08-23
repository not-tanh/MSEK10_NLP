import streamlit as st

from config import MODEL_PATH, DATA_PATH
from retriever import Retriever
from reader.predictor import Predictor, QuestionContextInput


@st.cache_resource
def load_resources():
    r = Retriever(DATA_PATH)
    p = Predictor(MODEL_PATH)
    return r, p


with st.spinner('Đang tải mô hình...'):
    retriever, predictor = load_resources()


with st.sidebar:
    st.title('💬 Simple Question Answering System')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hỏi tôi đi"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hỏi tôi đi"}]


st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

user_msg = st.chat_input('Nhập câu hỏi')
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message('user'):
        st.write(user_msg)

    documents = retriever.find_relevant_documents(user_msg, k=5)
    st.sidebar.write('Các tài liệu liên quan:')
    st.sidebar.write(documents)
    qci = QuestionContextInput(question=user_msg, context=documents[0]['context'])
    answer = predictor.answer([qci])
    print(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer['answer']})
    with st.chat_message('assistant'):
        st.write(answer['answer'])
