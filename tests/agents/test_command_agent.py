"""command_agent 的守卫（docs/20 F8）：运行不得污染 stdout。"""

from agents.command_agent import command_agent


def test_command_agent_ainvoke_writes_nothing_to_stdout(capsys) -> None:
    """生产路径不得往 stdout 打调试输出（此前有 print("Called A/B/C")）。

    反例注入：把 command_agent 里的 print 加回去，本用例必须变红。
    """
    from langchain_core.messages import HumanMessage

    command_agent.invoke({"messages": [HumanMessage(content="hi")]})
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
