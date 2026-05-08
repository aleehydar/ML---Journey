import os
from tavily import TavilyClient

def get_tavily_client():
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None
    return TavilyClient(api_key=api_key)

def pakistan_search(query: str) -> str:
    """
    Takes a query string, appends 'Pakistan' for local relevance,
    and returns top 5 results as clean text (title + snippet + url).
    Handles errors gracefully.
    """
    client = get_tavily_client()
    if not client:
        return "Search API Error: TAVILY_API_KEY is missing. Fallback to general knowledge."
        
    try:
        local_query = f"{query} in Pakistan"
        response = client.search(local_query, search_depth="basic", max_results=5)
        
        results = response.get('results', [])
        if not results:
            return "No relevant results found."
            
        formatted_results = []
        for i, res in enumerate(results):
            title = res.get('title', 'No Title')
            snippet = res.get('content', 'No Content')
            url = res.get('url', 'No URL')
            formatted_results.append(f"{i+1}. {title}\n   Snippet: {snippet}\n   URL: {url}")
            
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Search API Error: {str(e)}. Fallback to general knowledge."
