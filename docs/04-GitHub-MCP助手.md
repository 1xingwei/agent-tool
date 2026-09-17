# GitHub MCP 助手

GitHub MCP 助手是一个专用 agent，使用 GitHub MCP（Model Context Protocol，模型上下文协议）工具完成仓库管理与开发工作流。它基于 LangGraph 的 `create_react_agent` 构建，是 ReAct（Reasoning and Acting，推理与行动）模式的清晰实现。

**这个 agent 的定位是演示如何使用 MCP（Model Context Protocol）服务器与工具的 agent。**

## 功能

配置了 PAT 之后，[GitHub MCP server](https://github.com/github/github-mcp-server) 会提供多种工具。

- 仓库管理（创建、克隆、浏览）
- Issue 管理（创建、列出、更新、关闭）
- Pull Request 管理（创建、审查、合并）
- 分支管理（创建、切换、合并）
- 文件操作（读取、写入、搜索）
- 提交操作（创建、查看历史）

## 配置

要启用 GitHub MCP 助手，需要配置以下环境变量：

### 必填项

```bash
# GitHub Personal Access Token（GitHub MCP server 必填）
# 若不设置，GitHub MCP agent 将没有任何工具可用
GITHUB_PAT=your_github_personal_access_token_here
```

### 可选项

```bash
# GitHub MCP server 地址（默认 https://api.githubcopilot.com/mcp/）
MCP_GITHUB_SERVER_URL=https://api.githubcopilot.com/mcp/
```

## GitHub Personal Access Token

要使用 GitHub MCP 助手，需要一个具备相应权限的 GitHub Personal Access Token（PAT）。GitHub MCP server 提供了大量仓库管理工具，不同工具需要不同的 scope。

1. 进入 GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. 生成一个新 token，勾选以下 scope：
   - `repo`（私有仓库的完整控制权）——启用 `create_issue`、`create_pull_request`、`get_file_contents`、`list_commits` 等工具
   - `read:org`（读取组织与团队成员信息）——启用组织相关工具
   - `read:user`（读取用户资料）——启用用户资料工具
   - `user:email`（访问用户邮箱地址）——启用邮箱相关功能

**注意**：可用工具取决于你的 PAT scope。按上面推荐的 scope 配置，你就能使用大多数仓库管理工具，包括创建 issue、创建 pull request、读取文件内容、列出提交记录。

## 使用

配置完成后，GitHub MCP 助手会以 `github-mcp-agent` 的名字出现在服务中。

### 示例提示词

以下是可以对 GitHub MCP 助手使用的示例提示词（保留英文，它们是直接发给 agent 的实际输入）：

- **"Describe the JoshuaC215/agent-service-toolkit repository"**——展示仓库信息与 README 内容
- **"List the recent commits in this repository"**——显示最近的提交历史
- **"What files are in the src directory?"**——列出指定目录下的文件
- **"Show me the README file"**——显示仓库 README 内容
- **"Create a new issue titled 'Bug: Login not working' with the description 'The login form is not responding to user input'"**——创建一个新 issue
- **"What are the open issues in this repository?"**——列出未关闭的 issue
- **"Create a pull request from feature-branch to main with the title 'Add new feature'"**——创建一个 pull request
- **"Show me information about this repository"**——显示仓库详情
