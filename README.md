# Essay Writer Agent (LangGraph)

[English](#english) · [中文](#中文)

---

## English

A multi-step **Essay Writer Agent** built with [LangGraph](https://github.com/langchain-ai/langgraph). It plans an outline, researches the topic via [Tavily](https://tavily.com/), drafts an essay, critiques it, gathers more evidence, and revises — all as a stateful graph with checkpointing. A [Gradio](https://www.gradio.app/) UI is included so you can step through the run interactively.

This project is adapted from the DeepLearning.AI short course **"AI Agents in LangGraph"** (Lesson 6), updated to work with current LangChain / LangGraph versions.

### Architecture

```
        ┌─────────┐
        │ planner │
        └────┬────┘
             ▼
      ┌──────────────┐
      │ research_plan│  ── Tavily search
      └──────┬───────┘
             ▼
        ┌─────────┐
        │ generate│  ── draft essay
        └────┬────┘
             ▼
    revision_number > max?
       ├─ yes → END
       └─ no  → reflect
                  │
                  ▼
        ┌─────────────────┐
        │ research_critique│ ── Tavily search
        └────────┬────────┘
                 ▼
            (back to generate)
```

State (`AgentState`):
- `task` – user prompt
- `plan` – outline produced by the planner
- `draft` – current essay draft
- `critique` – teacher-style critique
- `content` – list of research snippets accumulated across iterations
- `revision_number`, `max_revisions` – iteration control

### Setup

```bash
# 1. Create venv & install deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Provide API keys in a .env file
cat > .env <<'EOF'
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
EOF
```

### Run

Open the notebook in VS Code / JupyterLab:

- `Le_6.ipynb` – step-by-step lesson notebook (build the graph, then `graph.stream(...)`)
- The bottom of `Le_6.ipynb` launches the Gradio GUI from `helper.py`:
  ```python
  from helper import ewriter, writer_gui
  MultiAgent = ewriter()
  app = writer_gui(MultiAgent.graph)
  app.launch()
  ```
  Open the printed `http://127.0.0.1:7860` URL in your browser.

### Files

| File | Purpose |
| --- | --- |
| `Le_6.ipynb` | Main lesson notebook – defines nodes, builds the graph, runs `stream()` |
| `helper.py` | `ewriter` class + `writer_gui` Gradio app (used by the notebook's last section) |
| `requirements.txt` | Pinned dependencies |
| `.env` | Your `OPENAI_API_KEY` and `TAVILY_API_KEY` (not committed) |

### Use OpenAI, not DeepSeek — known pitfalls

DeepSeek's API claims OpenAI-compatibility, but several LangChain/LangGraph features used in this project break against DeepSeek. **If you can, use OpenAI; if you must use DeepSeek, apply the workarounds below.**

#### 1. `with_structured_output(...)` fails with `BadRequestError`

The planner and research nodes use:
```python
model.with_structured_output(Queries).invoke([...])
```
Under the hood LangChain calls OpenAI's Structured Outputs API:
```json
"response_format": {"type": "json_schema", "json_schema": {...}}
```
DeepSeek **does not implement this**. You will see:
```
BadRequestError: Error code: 400 -
  {'error': {'message': 'This response_format type is unavailable now', ...}}
```
**Workarounds (DeepSeek only):**
- Pass `method="function_calling"` if your DeepSeek model supports tool calling:
  ```python
  model.with_structured_output(Queries, method="function_calling")
  ```
- Or drop `with_structured_output` and parse JSON yourself in the prompt:
  ```python
  resp = model.invoke([SystemMessage(content=PROMPT +
      '\n\nReturn ONLY a JSON object: {"queries": ["q1","q2","q3"]}.'),
      HumanMessage(content=task)])
  data = json.loads(re.search(r'\{.*\}', resp.content, re.DOTALL).group())
  queries = data["queries"]
  ```
With OpenAI's `gpt-4o-mini` / `gpt-4o` / `gpt-3.5-turbo`, none of this is needed — `with_structured_output` Just Works.

#### 2. `OPENAI_API_BASE` / `OPENAI_BASE_URL` leakage

People often set `OPENAI_API_BASE=https://api.deepseek.com` in `.env` to route OpenAI SDK calls to DeepSeek. If you later set `model="gpt-4o"` you'll hit `Model not found` because DeepSeek has no `gpt-*` models. Either:
- Unset the base URL when switching back to real OpenAI, or
- Pass `base_url=` explicitly per `ChatOpenAI(...)` instance.

#### 3. Embeddings / vision / tool-calling support is uneven

Outside this project's scope, but worth knowing: DeepSeek's coverage of OpenAI features (`response_format`, parallel tool calls, vision, embeddings models) varies by month. Code that worked against OpenAI may silently degrade on DeepSeek.

#### Recommended for this project

```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)   # cheap + supports structured output
```

### Troubleshooting (real issues we hit)

| Symptom | Root cause | Fix |
| --- | --- | --- |
| `KeyError: 'content'` in `research_plan_node` | Initial state passed to `graph.stream(...)` was missing `content` | Add `"content": []` to the initial state dict |
| `KeyError: 'content'` on `r['content']` | Some Tavily results don't carry `content` (only `raw_content`, or filtered) | `text = r.get('content') or r.get('raw_content') or ''` |
| `BadRequestError: response_format type is unavailable` | DeepSeek does not support `with_structured_output`'s json_schema mode | See section above — switch model or parse manually |
| `ModuleNotFoundError: langgraph.checkpoint.sqlite` | Optional package not installed | `pip install langgraph-checkpoint-sqlite` |
| `ModuleNotFoundError: langchain_core.pydantic_v1` | Removed in recent `langchain-core` | Change import to `from pydantic import BaseModel` |
| `ModuleNotFoundError: gradio` | Optional GUI dependency | `pip install gradio` |
| VS Code: edits to `.ipynb` keep getting reverted after re-running | VS Code's "Hot Exit" restores the in-memory buffer over your on-disk edits | Close the notebook tab and choose **Don't Save**, or open a renamed copy of the file |
| Gradio UI: all Textboxes show red "错误" / "Error" badges, but `Live Agent Output` shows the *whole tuple* | Gradio 6.x strictly checks output counts — `helper.py` had `outputs=[live]` on `gen_btn.click(...)` but `run_agent` yields 6 values, so the whole tuple is shoved into `live` and the other 5 boxes get nothing | Bind all 6 outputs: `outputs=[live, lnode_bx, nnode_bx, threadid_bx, revision_bx, count_bx]` for both `gen_btn` and `cont_btn` |
| Gradio UI: a Textbox shows a red "错误" / "Error" badge even though a value is being written | Gradio 6.x rejects non-`str` values for `Textbox.value` (int, tuple, etc.). Old gradio coerced silently | Wrap with `str(...)` when yielding / returning values for Textboxes |
| `KeyError: 'thread_ts'` inside `helper.py` | LangGraph renamed `config['configurable']['thread_ts']` to `checkpoint_id` in recent versions | Use `config['configurable'].get('checkpoint_id') or config['configurable'].get('thread_ts')` for back-compat |

---

## 中文

基于 [LangGraph](https://github.com/langchain-ai/langgraph) 的多步骤 **Essay Writer Agent**(论文写作 Agent)。它会先生成大纲、用 [Tavily](https://tavily.com/) 搜索资料、起草论文、自我批评、再次检索、然后修订 —— 整个过程是一个带 checkpoint 的有状态图。项目自带一个 [Gradio](https://www.gradio.app/) 界面,可以交互式地一步一步看 Agent 怎么思考。

本项目改编自 DeepLearning.AI 短课程 **"AI Agents in LangGraph"** 第 6 课,并适配了当前版本的 LangChain / LangGraph。

### 架构

```
        ┌─────────┐
        │ planner │  生成大纲
        └────┬────┘
             ▼
      ┌──────────────┐
      │ research_plan│  Tavily 检索
      └──────┬───────┘
             ▼
        ┌─────────┐
        │ generate│  写草稿
        └────┬────┘
             ▼
   revision_number > max?
      ├─ 是 → 结束
      └─ 否 → reflect (批评)
                 │
                 ▼
       ┌─────────────────┐
       │ research_critique│  再检索
       └────────┬────────┘
                ▼
           (回到 generate)
```

State (`AgentState`) 字段:
- `task` – 用户输入题目
- `plan` – planner 给出的大纲
- `draft` – 当前论文草稿
- `critique` – 教师视角的批评
- `content` – 历次研究检索累积的文本片段
- `revision_number`, `max_revisions` – 迭代控制

### 安装

```bash
# 1. 建虚拟环境 + 装依赖
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置 API key
cat > .env <<'EOF'
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
EOF
```

### 运行

在 VS Code 或 JupyterLab 里打开:

- `Le_6.ipynb` – 主 notebook:逐步搭建 graph,然后 `graph.stream(...)` 跑一次
- notebook 末尾启动 Gradio 界面:
  ```python
  from helper import ewriter, writer_gui
  MultiAgent = ewriter()
  app = writer_gui(MultiAgent.graph)
  app.launch()
  ```
  浏览器打开输出里的 `http://127.0.0.1:7860`。

### 文件说明

| 文件 | 作用 |
| --- | --- |
| `Le_6.ipynb` | 主 notebook,定义节点、构建 graph、跑 stream |
| `helper.py` | `ewriter` 类 + Gradio 界面(notebook 末尾用) |
| `requirements.txt` | 锁定的依赖版本 |
| `.env` | 你自己的 `OPENAI_API_KEY` 和 `TAVILY_API_KEY`,不应提交 |

### 推荐用 OpenAI,不推荐用 DeepSeek —— 实测踩过的坑

DeepSeek 的 API 号称兼容 OpenAI 接口,但本项目用到的几个 LangChain/LangGraph 特性在 DeepSeek 上是**坏的**。**如果有 OpenAI key,直接用 OpenAI;非要用 DeepSeek,看下面的绕过办法。**

#### 1. `with_structured_output(...)` 直接报 `BadRequestError`

planner 和 research 节点都这么写:
```python
model.with_structured_output(Queries).invoke([...])
```
LangChain 在底层用的是 OpenAI 的 Structured Outputs API:
```json
"response_format": {"type": "json_schema", "json_schema": {...}}
```
**DeepSeek 不实现这个**。你会看到:
```
BadRequestError: Error code: 400 -
  {'error': {'message': 'This response_format type is unavailable now', ...}}
```
**绕过办法(只在 DeepSeek 下需要):**
- 如果 DeepSeek 模型支持 tool calling,可以用 `function_calling` 模式:
  ```python
  model.with_structured_output(Queries, method="function_calling")
  ```
- 或者干脆不用 `with_structured_output`,自己 prompt 让模型返回 JSON 然后手动解析:
  ```python
  resp = model.invoke([SystemMessage(content=PROMPT +
      '\n\n仅返回 JSON 对象: {"queries": ["q1","q2","q3"]},不要别的文字。'),
      HumanMessage(content=task)])
  data = json.loads(re.search(r'\{.*\}', resp.content, re.DOTALL).group())
  queries = data["queries"]
  ```
用 OpenAI 的 `gpt-4o-mini` / `gpt-4o` / `gpt-3.5-turbo`,这些麻烦都没有 —— `with_structured_output` 开箱即用。

#### 2. `OPENAI_API_BASE` / `OPENAI_BASE_URL` 串台

很多人为了让 OpenAI SDK 调 DeepSeek,会在 `.env` 里设 `OPENAI_API_BASE=https://api.deepseek.com`。之后你切回 OpenAI、写 `model="gpt-4o"` 时,会撞上 `Model not found`(DeepSeek 没有 `gpt-*` 模型)。两个办法:
- 切回 OpenAI 时把这个环境变量取消,或者
- 每个 `ChatOpenAI(...)` 实例显式传 `base_url=`。

#### 3. embedding / 视觉 / tool-calling 覆盖度不齐

这个项目没用到,但顺便提醒:DeepSeek 对 OpenAI 周边能力(`response_format`、并行 tool call、视觉、embedding 模型)的支持月月在变,曾经能跑的代码,可能某天就静默退化。

#### 本项目的推荐配置

```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)   # 便宜 + 完整支持 structured output
```

### 排错对照表(本次实际踩到的坑)

| 现象 | 根因 | 修复 |
| --- | --- | --- |
| `research_plan_node` 抛 `KeyError: 'content'` | `graph.stream(...)` 的初始 state 缺少 `content` 字段 | 在初始 state 加 `"content": []` |
| `r['content']` 抛 `KeyError: 'content'` | Tavily 返回的某个 result 没有 `content`(只有 `raw_content`,或被过滤) | 改成 `text = r.get('content') or r.get('raw_content') or ''` |
| `BadRequestError: response_format type is unavailable` | DeepSeek 不支持 `with_structured_output` 的 json_schema 模式 | 见上面 "DeepSeek 的坑" 章节,换模型或手动解析 |
| `ModuleNotFoundError: langgraph.checkpoint.sqlite` | 包没装 | `pip install langgraph-checkpoint-sqlite` |
| `ModuleNotFoundError: langchain_core.pydantic_v1` | 新版 `langchain-core` 删了这个兼容层 | 改成 `from pydantic import BaseModel` |
| `ModuleNotFoundError: gradio` | GUI 依赖没装 | `pip install gradio` |
| VS Code 里 `.ipynb` 怎么改都没生效 / 改完又被回滚 | VS Code 的 Hot Exit 会用内存里的旧 buffer 覆盖磁盘 | 关 tab 时点 **"Don't Save"**,或者把文件改名再打开 |
| Gradio 界面所有 Textbox 显示红框"错误",但 `Live Agent Output` 里能看到**完整的 tuple** | Gradio 6.x 严格校验输出数量 —— `helper.py` 里 `gen_btn.click(...)` 只绑了 `outputs=[live]`,但 `run_agent` yield 6 个值,新版直接把整个 tuple 塞给 `live`,其余 5 个 Textbox 拿不到值 | 把 outputs 列全:`outputs=[live, lnode_bx, nnode_bx, threadid_bx, revision_bx, count_bx]`(`gen_btn` 和 `cont_btn` 都要) |
| Gradio 界面某个 Textbox 红框"错误",即使后端确实在写值 | Gradio 6.x 不再静默接受非 `str` 类型(int / tuple 等)作为 `Textbox.value`,老 gradio 会自动 cast | 在 yield / return 给 Textbox 的值外面包一层 `str(...)` |
| `helper.py` 抛 `KeyError: 'thread_ts'` | 新版 LangGraph 把 `config['configurable']['thread_ts']` 改名为 `checkpoint_id` | 用 `config['configurable'].get('checkpoint_id') or config['configurable'].get('thread_ts')` 兼容新旧两版 |
