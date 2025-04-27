import re
from docx import Document
from transformers import AutoTokenizer
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

### Считаем текст с книги в формате docx
doc = Document('Руцкая Тамара. Полный справочник симптомов.docx')

### Сохраним весь текст в переменной text
text = []

for par in doc.paragraphs:
  text.append(par.text)

### Глазами отчленим ненужные данные из книги и будем работать с нужными
text = text[31:-3]
str_text = ' '.join(text)

def clean_text(text):
  # Удалим лишние пробелы и переносы строк
  text = re.sub(r'\s+', ' ', text).strip()

  return text

cleaned_text = clean_text(str_text)

# emb_name = 'ai-forever/rugpt3small_based_on_gpt2'
emb_name = 'BAAI/bge-m3'

# создадим сплиттер с нашим эмбеддером
text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    AutoTokenizer.from_pretrained(emb_name),
    chunk_size=1000, # размер чанков
    chunk_overlap=100, # пересечение чанков
    separator=' ' # разделитель
)

texts = text_splitter.split_text(cleaned_text)
# сохраним эмбеддинги чанков в векторной бд
embedding_model = HuggingFaceEmbeddings(
    model_name=emb_name,
    multi_process=True,
    # model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
)

KNOWLEDGE_VECTOR_DATABASE = FAISS.from_texts(
    texts, embedding_model, distance_strategy=DistanceStrategy.COSINE
)
KNOWLEDGE_VECTOR_DATABASE.save_local('Руцкая_store_index')