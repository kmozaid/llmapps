from typing import List

import dspy
from dotenv import load_dotenv

DOCS = {}

def search(query: str, k: int) -> list[str]:
    results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(
        query, k=k
    )
    results = [x["text"] for x in results]

    for result in results:
        title, text = result.split(" | ", 1)
        DOCS[title] = text

    return results


def search_wikipedia(query: str) -> list[str]:
    """Returns top-5 results and then the titles of the top-5 to top-30 results."""
    print(f"Entering search_wikipedia {query=}")

    topK = search(query, 30)
    titles, topK = [f"`{x.split(' | ')[0]}`" for x in topK[5:30]], topK[:5]
    return topK + [f"Other retrieved pages have titles: {', '.join(titles)}."]


def lookup_wikipedia(title: str) -> str:
    """Returns the text of the Wikipedia page, if it exists."""
    print(f"Entering lookup_wikipedia {title=}")

    if title in DOCS:
        return DOCS[title]

    results = [x for x in search(title, 10) if x.startswith(title + " | ")]
    if not results:
        return f"No Wikipedia page found for title: {title}"
    return results[0]


class ClaimTitles(dspy.Signature):
    """Find all Wikipedia titles relevant to verifying (or refuting) the claim."""
    claim: str = dspy.InputField(desc="The claim.")
    titles: List[str] = dspy.OutputField(desc="The associated titles.")

if __name__ == "__main__":
    load_dotenv()
    print("Hello ReAct DSPy!!")

    lm = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=lm)

    #instructions = ("Find all Wikipedia titles relevant to verifying (or refuting) the claim."
    #signature = dspy.Signature("claim -> titles: list[str]", instructions)

    react = dspy.ReAct(
        ClaimTitles, tools=[search_wikipedia, lookup_wikipedia], max_iters=10
    )

    res = react(claim="David Gregory was born in 1625.")

    print(res)

    print(res.titles[:3])
