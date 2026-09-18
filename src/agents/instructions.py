"""跨 agent 共享的提示词常量。

`SEARCH_STOP_CONDITION` 收敛 loop_agent 与 research_assistant 两处曾各自写的
搜索止损条款 —— 两处副本正是「改一处漏一处」bug 的成因（docs/11 §1.1 P1）。
"""

SEARCH_STOP_CONDITION = """
Search efficiently and stop early. Two or three web searches on the same question are enough:
if the results so far do not contain the answer, tell the user plainly what you could not find
instead of rewording the query and searching again. Never repeat a search that returned no new
information, and do not keep searching just to fill a gap you already know the results miss.
"""
