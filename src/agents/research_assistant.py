from datetime import datetime

from langchain_community.tools import OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper

from agents.graph_factory import build_tool_agent_graph
from agents.tools import calculator, web_search
from core import settings

tools = [web_search, calculator]

# 如果设置了 API key，则添加天气工具
# 在 https://openweathermap.org/api/ 注册获取 API key
if settings.OPENWEATHERMAP_API_KEY:
    wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=wrapper))

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
    - Search efficiently and stop early. Two or three web searches on the same question are enough:
      if the results so far do not contain the answer, tell the user plainly what you could not find
      instead of rewording the query and searching again. Never repeat a search that returned no new
      information, and do not keep searching just to fill a gap you already know the results miss.
    """


research_assistant = build_tool_agent_graph(tools=tools, instructions=instructions)
