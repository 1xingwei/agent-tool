# 创建 RAG 检索助手

你可以用 Chroma 数据库构建一个 RAG 检索助手。

## 准备 Chroma 数据库

创建 Chroma 数据库的步骤：

1. 把要用的数据放进一个目录，例如 `./data`。支持的格式：`.md`、`.txt`、`.pdf`、`.docx`。
2. 打开 [`create_chroma_db.py`](../scripts/create_chroma_db.py)，把 folder_path 变量设为你的数据目录，例如 `./data`。
3. 可以修改数据库名、分块大小（chunk size）与重叠大小（overlap size）。数据库路径默认为 `settings.CHROMA_DIR`（`.env` 中的 `CHROMA_DIR` 键），缺省回退到 `./var/chroma_db`。
4. 假定你已经完成[快速开始](../README.md#quickstart)并激活了虚拟环境，运行以下命令建库：

   ```sh
   python scripts/create_chroma_db.py
   ```

   重复运行是安全的——除非传入 `delete_chroma_db=True`，否则已有数据不会被删除。

5. 建库成功后，Chroma 数据库会生成在 `var/` 下（存放本机运行时产物的目录）。

## 配置 RAG 检索助手

创建一个 RAG 检索助手：

1. 把 `settings.CHROMA_DIR`（或环境变量 `CHROMA_DIR`）指向你刚建好的数据库；`src/agents/tools.py` 在加载 retriever 时会读它。
2. 调整返回的文档数量，当前设为 5。
3. 更新 `database_search_func` 函数的描述，准确说明你的数据库用途与内容。
4. 打开 [`rag_assistant.py`](../src/agents/rag_assistant.py)，修改 agent 的 instructions，说明这个助手的专长是什么、能访问哪些知识，例如：

   ```python
   instructions = f"""
       You are a helpful HR assistant with the ability to search a database containing information on our company's policies, benefits and handbook.
       Today's date is {current_date}.

       NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

       A few things to remember:
       - If you have access to multiple databases, gather information from a diverse range of sources before crafting your response.
       - Please include the source of the information used in your response.
       - Use a friendly but professional tone when replying.
       - Only use information from the database. Do not use information from outside sources.
       """
   ```

5. 打开 [`streamlit_app.py`](../src/streamlit_app.py)，修改 agent 的欢迎语：

   ```python
   WELCOME = """Hello! I'm your AI-powered HR assistant, here to help you navigate company policies, the employee handbook, and benefits. Ask me anything!"""
   ```

6. 运行应用，测试你的 RAG 检索助手。
