import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.schema import HumanMessage
from get_RAG_models import get_qa_answer
from get_models import model

class State(TypedDict):
    is_medical: str
    question: str
    rag_answer: str


def classification_node(state: State):
    '''
     Классифицирует текст: связан с медициной или нет
    :param state:
    :return:
    '''

    prompt = PromptTemplate(
        input_variables=['question'],
        template='Классифицируй текст, связан ли он с медициной, или нет. Дай ответ, связан или не связан с медициной. Текст: {text}'
    )

    message = HumanMessage(content=prompt.format(text=state['question']))
    is_medical = model.invoke([message]).content.strip()

    return {'is_medical': is_medical}

def rag_node(state: State):
    '''
        Берет информацию из базы знаний по симптомам и отвечает по ней, если находит подходящую информацию
    :param state:
    :return:
    '''

    return {'answer': get_qa_answer(State['question'])}


workflow = StateGraph(State)

# Добавление узлов
workflow.add_node('is_medical', classification_node)
workflow.add_node('rag_answer', rag_node)

# Обозначим границы узлов
workflow.set_entry_point("is_medical") # Вход в граф
workflow.add_edge("is_medical", "rag_answer")
workflow.add_edge("rag_answer", END)

app = workflow.compile()

state_input = {'question': 'Покраснение глаза и боль в нем.'}

result = app.invoke(state_input)

print(result)