import openai
from config import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
def gpt3(prompt_, stop=None):
    #if stop is None:
    #    stop = ["\n"]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[{"role": "user", "content": prompt_}],
        temperature=0,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop
    )
    return response["choices"][0]["message"]["content"]

def gpt4(prompt_, stop=None):
    #if stop is None:
    #    stop = ["\n"]
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",#"gpt-4",
        messages=[{"role": "user", "content": prompt_}],
        temperature=0,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop
    )
    return response["choices"][0]["message"]["content"]
