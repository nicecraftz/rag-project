import ollama

from service.env_provider import ENRICHMENT_MODEL


def enrich_query(query: str) -> str:
    PROMPT = """
    Objective:
        1. You are a query enrichment agent, your role is to enrich the users query by replying to them as if you were a helpdesk agent.
        2. You should try to contextualize more the user's intent as their query could be very small, you should expand it.
        3. You should always reply in english

    Constraints:
        - Never go longer than 200 words
        - Never add extra comments
    """
    response = ollama.chat(
        model=ENRICHMENT_MODEL, messages=[system(PROMPT), user(query)]
    )

    return response["message"]["content"]


def enrich_response(query: str, chunks: list[str]) -> str:
    PROMPT = """
        Objective:
            1. You are a agent that is given the task to create a structured and easy response for the user
            2. Your responses must be based out of the chunks context and user query, never generate anything out of your pocket.

        Rules:
            - You MUST reply using the user's language (italian if the user is italian, english if the user is english and so on.)
            - Never add extra comments, you must reply as if you were a real human help-agent, don't structure messages.
    """
    response = ollama.chat(
        model=ENRICHMENT_MODEL,
        messages=[
            system(PROMPT),
            user("\n\n".join(chunks)),
            user("USER QUERY: " + query),
        ],
    )

    return response["message"]["content"]


def system(prompt: str):
    return {"role": "system", "content": prompt}


def user(prompt: str):
    return {"role": "user", "content": prompt}
