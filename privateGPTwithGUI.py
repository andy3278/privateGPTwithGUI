
#!/usr/bin/env python3
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def main():
    st.title("PrivateGPT")

    # upload file
    st.header("Upload any document")
    uploaded_file = st.file_uploader("Choose a file")

    # enter query
    st.header("enter query: ")
    query = st.text_input("")

    # get answer
    if st.button("get answer"):
        if uploaded_file is None:
            st.warning("Please upload a file.")
        elif query == "":
            st.warning("Please enter a query.")
        else:
            embeddings = HuggingFaceEmbeddings(model_name = embeddings_model_name)
            db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
            retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
            # activate or deactivate streaming stdout callback for LLM
            callbacks = [] if True else [StreamingStdOutCallbackHandler()]
            # get LLM model
            match model_type :
                case "LlamaCpp":
                    llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
                    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents = True)
                case "GPT4ALL":
                    llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
                    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents = True)
                case _:
                    print(f"Model {model_type} not supported.")
                    exit;
            
            # get answer from chain
            res = qa(query)
            answer, docs = res["result"], res['source_documents']

            # print result
            st.header("Result: ")
            st.write(f"Query: {query}")
            st.write(f"Answer: {answer}")

            # print source docuemnt if exist
            if docs is not None:
                st.header("Source documents: ")
                for doc in docs:
                    st.subheader(doc.metadata["source"])
                    st.write(doc.page_content)

if __name__ == "__main__":
    main()
