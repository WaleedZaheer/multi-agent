"""
tools.py
--------
A "tool" is just a normal Python function an agent calls to interact with the
outside world. This one does a free web search (no API key required) using
the duckduckgo_search library directly.

Note: we're NOT using LangChain's DuckDuckGoSearchRun wrapper here - that
wrapper lives in langchain-community, which is being phased out. Calling the
search library directly is simpler and won't break when that package changes.
"""

from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 5) -> str:
    """
    Searches the web and returns a compact, readable block of results.
    If the search fails (rate limit, no internet, etc.) it returns a
    clearly-labelled error string instead of crashing the whole pipeline.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"[web search failed: {e}]"

    if not results:
        return "[no search results found]"

    formatted = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        formatted.append(f"{i}. {title}\n   {body}\n   source: {href}")

    return "\n\n".join(formatted)
