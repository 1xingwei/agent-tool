from datetime import datetime

from agents.graph_factory import build_tool_agent_graph
from agents.instructions import SEARCH_STOP_CONDITION
from agents.tools import calculator, fetch_url, weather, web_search

tools = [web_search, calculator, fetch_url, weather]

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a helpful research assistant with the ability to search the web and use other tools.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Use calculator tool with numexpr to answer math questions. The user does not understand numexpr,
      so for the final response, use human readable format - e.g. "300 * 200", not "(300 \\times 200)".
    {SEARCH_STOP_CONDITION}
    - Tool order: first use WebSearch to find candidate pages; when a snippet is too short to contain
      the answer, open the page body with fetch_url; stop once you have it.
    """


research_assistant = build_tool_agent_graph(tools=tools, instructions=instructions)
