#!/usr/bin/env python
#  -*- coding: utf-8 -*-
__date__ = '2023/2/25'
import openai
import numpy as np
import pandas as pd
from openai.embeddings_utils import distances_from_embeddings

# Set the API key
openai.api_key = ''

# Load the dataframe
df = pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
df.head()


def create_context(
        question, df, max_len=100000000000, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
        df,
        model="text-davinci-003",
        question="Am I allowed to publish model outputs to Twitter, without a human review?",
        max_len=1800,
        size="ada",
        debug=False,
        max_tokens=600,
        stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"解答下面关于 tqsdk 的相关问题, 如果无法回答, 请回答 \"抱歉, 我不了解\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=1,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


if __name__ == '__main__':
    while True:
        user_input = input("Input your question (or 'quit' to exit): ")
        if user_input == "quit":
            break
        result = answer_question(df, question=user_input, debug=False)
        print(result)
