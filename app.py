"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake


st.set_page_config(page_title="Twitterアルゴさん", page_icon="🐦")
st.header("🐦 Twitterアルゴさん")

is_gpt4 = st.checkbox('Enable GPT4',help="With this it might get slower")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


api_token = os.environ['OPENAI_API_KEY'] # st.text_input('OpenAI API Token',type="password")

def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAI:{ai}")
    return "\n".join(res)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    if is_gpt4:
        model = "gpt-4"
    else:
        model = "gpt-3.5-turbo"
    llm = ChatOpenAI(temperature=0.9, model_name=model, streaming=True, verbose=True)
    embeddings = OpenAIEmbeddings(disallowed_special=())
    db = DeepLake(dataset_path="hub://davitbun/twitter-algorithm", read_only=True, embedding_function=embeddings)
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 10
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever,memory=memory)
    return chain

    
def get_text():
    input_text = st.text_input("You: ", "SimClustersへの割り当てはどのように行うのですか？", key="input")
    return input_text

with st.form(key='ask'):
    user_input = get_text()
    ask_button = st.form_submit_button('ask')

if ask_button:
    with st.spinner('typing...'):
        chat_history = []
        qa = load_chain()
        prefix = 'あなたはTwitterのアルゴリズムについて詳しいマーケティングの専門家です。下記の質問に対して初心者にもわかりやすく解説してください。知識にないものはわからないとこたえてください。\n質問:'
      # 'you are a very helpful explainer of videos. The attached is a transcript of a YouTube video and your task is to answer question. if you dont have a good answer based on the video, please say you do not know. yo your answer should be the same as i use after this sentence.  '
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result['answer']))
        st.session_state.past.append(user_input)
        st.session_state.generated.append(result['answer'])


if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        try:
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        except:
            pass
