# imports
import wikipedia
from openai import OpenAI
from IPython.display import Markdown, display
# constants
MODEL_LLAMA = 'llama3.2'
OLLAMA_BASE_URL = "http://localhost:11434/v1"

ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key="anything")


# set up environment


def explain_term(term, sentences=3):
    try:
        # Set language explicitly (optional)
        wikipedia.set_lang("en")

        # Get a summary of the term
        summary = wikipedia.summary(term, sentences=sentences)
        return summary

    except wikipedia.exceptions.DisambiguationError as e:
        return f"Your query is ambiguous. Did you mean one of these?\n{e.options[:5]}"

    except wikipedia.exceptions.PageError:
        return "Sorry, I could not find information on that term."

    except Exception as e:
        return f"An unexpected error occurred: {e}"


# here is the question; type over this to ask something new

system = """
Please explain what this particular sentence means in a simple and helpful language, as if I am your friend and can understand things better only when you explain it.
"""

# Get Llama 3.2 to answer
def answer_from_wiki(term):
    ans = ollama.chat.completions.create(model = MODEL_LLAMA, 
                                         messages = [
                                             {"role": "system", "content":system}, 
                                             {"role":"user", "content": term }
                                         ])
    return ans.choices[0].message.content

