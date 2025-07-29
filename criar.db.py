#carrega os arquivos pdf em texto
from langchain_community.document_loaders import PyPDFDirectoryLoader
#divide os documentos em chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
#cria o banco de dados apartir de chunks de documentos
from langchain_chroma.vectorstores import Chroma
#modelo de embeddings
from langchain_openai import OpenAIEmbeddings
#carrega o .env como variavel de ambiente 
from dotenv import load_dotenv

load_dotenv()

PASTA_BASE = "base"

def criar_db():
    documentos = carregar_documentos()
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)

#ler os documentos em pdf da pasta "base"
def carregar_documentos():
    carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="*.pdf")
    documentos = carregador.load()
    return documentos

#paremetros das chunks
def dividir_chunks(documentos):
    separador_documentos = RecursiveCharacterTextSplitter( 
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
     )
    chunks = separador_documentos.split_documents(documentos)
    print(len(chunks))
    return chunks


def vetorizar_chunks(chunks):
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="db")
    print("banco de dados criado")



criar_db()