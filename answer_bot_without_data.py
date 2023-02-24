import openai

openai.api_key = ''

def answer_question(question):
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"请回答下面关于 tqsdk 的问题, {question}",
        temperature=1,
        max_tokens=600,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0,
        stop=[" Human:", " AI:"]
    )
    return response.choices[0].text

if __name__ == '__main__':
    while True:
        user_input = input("Input your question (or 'quit' to exit): ")
        if user_input == "quit":
            break
        result = answer_question(question=user_input)
        print(result)