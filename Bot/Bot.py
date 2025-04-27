import os

from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv, find_dotenv
from get_RAG_models import get_qa_answer


load_dotenv(find_dotenv())

bot = Bot(os.environ['TELEGRAM_TOKEN'])  # создание бота
dp = Dispatcher(bot)  # анализ и инициализация всех входящих апдейтов, функционал бота


@dp.message_handler(commands='start')
async def echo(payload: types.Message):
    await payload.reply('Привет')

@dp.message_handler(commands='start')
async def echo(payload: types.Message):
    await payload.reply('Привет')


if __name__ == '__main__':
    executor.start_poolling(dp)

