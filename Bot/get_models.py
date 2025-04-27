from yandex_cloud_ml_sdk import YCloudML
from langchain_community.llms import YandexGPT
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(usecwd=True))
def ya_model():
    model = YandexGPT(
        iam_token=YANDEX_TOKEN,
        folder_id=YANDEX_FOLDER_ID,
        template=0
    )

    return model

model = ya_model()