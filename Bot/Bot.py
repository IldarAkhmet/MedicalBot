from aiogram import Bot, Dispatcher, executor, types
from TOKEN import TOKEN


bot = Bot(TOKEN)  # создание бота
dp = Dispatcher(bot)  # анализ и инициализация всех входящих апдейтов, функционал бота


@dp.message_handler(commands='start')
async def echo(payload: types.Message):
    await payload.reply('Привет')



if __name__ == '__main__':
    executor.start_poolling(dp)

