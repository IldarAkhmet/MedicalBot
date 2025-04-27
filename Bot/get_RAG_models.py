from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from prompts import prompt_template
from get_models import model
import os

class GlobalParameters:
    def __init__(self):
        self.emb_name = 'BAAI/bge-m3'
        self.vector_data_path = 'Data/Руцкая_store_index'
gp = GlobalParameters()

embedding_model = HuggingFaceEmbeddings(
    model_name=gp.emb_name,
    multi_process=True,
    # model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
)

vector_db = FAISS.load_local(
    folder_path=gp.vector_data_path,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_llm(
    llm=model,
    retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

qa_chain.combine_documents_chain.llm_chain.prompt = PROMPT


def get_qa_answer(query, chain=qa_chain):
    result = qa_chain.invoke({'query': query})

    print('Использованные чанки:', [doc.page_content[:50] + "..." for doc in result["source_documents"]])
    return result['result']
