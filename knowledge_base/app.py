from flask import Flask, request, jsonify, Response, render_template_string
from openai import OpenAI
import json
import time
from collections import deque

# ── 如果没有 search_knowledge 模块，用下面的 stub 代替 ──
try:
    from search_knowledge import rrf_fusion
except ImportError:
    def rrf_fusion(query):
        return [f"[search_knowledge 未加载，无法检索 '{query}' 的相关内容]"]
'''
支持多轮对话，对话轮数参数max_turns
支持流失回答。
'''

app = Flask(__name__)

# ── 全局对话状态（多用户可改为 session/Redis 存储）──
class DialogueStore:
    def __init__(self, max_turns=5):
        self.client = OpenAI(
            base_url='http://192.168.0.1:5156/v1',
            api_key='none'
        )
        self.max_turns = max_turns
        self.history: deque = deque(maxlen=max_turns)

    def build_messages(self, current_query: str):
        messages = []
        for turn in self.history:
            messages.append({'role': 'user',      'content': turn['user']})
            messages.append({'role': 'assistant', 'content': turn['assistant']})
        messages.append({'role': 'user', 'content': current_query})
        return messages

    def add_history(self, user_msg: str, assistant_msg: str):
        self.history.append({'user': user_msg, 'assistant': assistant_msg})

    def clear(self):
        self.history.clear()

    def get_history(self):
        return list(self.history)

dialogue_store = DialogueStore(max_turns=5)

# ────────────────────────────────────────────────
# 前端 HTML（单文件，内联 CSS/JS）
# ────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>知识问答系统</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #0d0f14;
    --surface:   #161922;
    --border:    #252a38;
    --accent:    #e8c97e;
    --accent2:   #7eb8e8;
    --text:      #d4cfc5;
    --text-dim:  #6b7280;
    --user-bg:   #1c2235;
    --ai-bg:     #161d2b;
    --radius:    12px;
    --font-ui:   'Noto Serif SC', serif;
    --font-mono: 'JetBrains Mono', monospace;
  }

  html, body { height: 100%; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-ui);
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  /* ── 顶栏 ── */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 28px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    flex-shrink: 0;
    gap: 12px;
  }
  .logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: .04em;
    color: var(--accent);
  }
  .logo svg { flex-shrink: 0; }
  .header-actions {
    display: flex;
    gap: 8px;
    align-items: center;
  }
  .turns-badge {
    font-family: var(--font-mono);
    font-size: .72rem;
    color: var(--text-dim);
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 10px;
  }
  .btn-clear {
    font-family: var(--font-ui);
    font-size: .78rem;
    color: var(--text-dim);
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 14px;
    cursor: pointer;
    transition: color .2s, border-color .2s;
  }
  .btn-clear:hover { color: #e07070; border-color: #e07070; }

  /* ── 聊天区 ── */
  #chat {
    flex: 1;
    overflow-y: auto;
    padding: 24px 0;
    scroll-behavior: smooth;
  }
  #chat::-webkit-scrollbar { width: 4px; }
  #chat::-webkit-scrollbar-track { background: transparent; }
  #chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  .msg-row {
    display: flex;
    padding: 0 20px;
    margin-bottom: 4px;
    animation: fadeUp .25s ease both;
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .msg-row.user  { justify-content: flex-end; }
  .msg-row.ai    { justify-content: flex-start; }

  .bubble {
    max-width: min(640px, 82vw);
    padding: 12px 16px;
    border-radius: var(--radius);
    font-size: .92rem;
    line-height: 1.75;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .msg-row.user .bubble {
    background: var(--user-bg);
    border: 1px solid #2d3650;
    color: #c7d4f0;
    border-bottom-right-radius: 4px;
  }
  .msg-row.ai .bubble {
    background: var(--ai-bg);
    border: 1px solid var(--border);
    color: var(--text);
    border-bottom-left-radius: 4px;
  }

  .avatar {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: .7rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .msg-row.user .avatar { order: 2; margin-left: 8px; background: #2d3650; color: var(--accent); }
  .msg-row.ai   .avatar { margin-right: 8px; background: #1c2535; color: var(--accent2); }

  /* 光标闪烁 */
  .cursor::after {
    content: '▍';
    color: var(--accent);
    animation: blink .6s step-end infinite;
  }
  @keyframes blink { 50% { opacity: 0; } }

  /* 欢迎提示 */
  #welcome {
    text-align: center;
    padding: 60px 20px 20px;
    color: var(--text-dim);
    font-size: .88rem;
    line-height: 2;
    pointer-events: none;
  }
  #welcome strong { color: var(--accent); font-size: 1.15rem; display: block; margin-bottom: 8px; }

  /* ── 输入区 ── */
  footer {
    border-top: 1px solid var(--border);
    background: var(--surface);
    padding: 14px 20px 18px;
    flex-shrink: 0;
  }
  .input-row {
    display: flex;
    gap: 10px;
    max-width: 860px;
    margin: 0 auto;
  }
  #inputArea {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text);
    font-family: var(--font-ui);
    font-size: .9rem;
    padding: 10px 16px;
    resize: none;
    min-height: 44px;
    max-height: 140px;
    line-height: 1.6;
    transition: border-color .2s;
    overflow-y: auto;
  }
  #inputArea:focus { outline: none; border-color: var(--accent); }
  #inputArea::placeholder { color: var(--text-dim); }

  .toggle-rag {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: .78rem;
    color: var(--text-dim);
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }
  .toggle-rag input { accent-color: var(--accent); cursor: pointer; }

  #sendBtn {
    background: var(--accent);
    color: #1a1400;
    border: none;
    border-radius: var(--radius);
    width: 44px;
    height: 44px;
    flex-shrink: 0;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: opacity .2s, transform .1s;
  }
  #sendBtn:hover:not(:disabled) { opacity: .88; }
  #sendBtn:active:not(:disabled) { transform: scale(.94); }
  #sendBtn:disabled { opacity: .35; cursor: not-allowed; }

  .hint { font-size: .72rem; color: var(--text-dim); text-align: center; margin-top: 8px; }
</style>
</head>
<body>

<header>
  <div class="logo">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
    </svg>
    知识问答系统
  </div>
  <div class="header-actions">
    <span class="turns-badge" id="turnsBadge">0 / 5 轮</span>
    <button class="btn-clear" onclick="clearHistory()">清空对话</button>
  </div>
</header>

<div id="chat">
  <div id="welcome">
    <strong>欢迎使用知识问答系统</strong>
    输入您的问题，系统将从知识库中检索相关内容并作答<br/>
    支持多轮连续对话，最多保留 5 轮上下文
  </div>
</div>

<footer>
  <div class="input-row">
    <textarea id="inputArea" rows="1" placeholder="输入问题，按 Enter 发送（Shift+Enter 换行）…"></textarea>
    <label class="toggle-rag">
      <input type="checkbox" id="ragToggle" checked/> 知识检索
    </label>
    <button id="sendBtn" onclick="sendMessage()" title="发送">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M2 21l21-9L2 3v7l15 2-15 2v7z"/></svg>
    </button>
  </div>
  <p class="hint">Enter 发送 · Shift+Enter 换行 · 可关闭「知识检索」进行纯对话</p>
</footer>

<script>
const chatEl   = document.getElementById('chat');
const inputEl  = document.getElementById('inputArea');
const sendBtn  = document.getElementById('sendBtn');
const badge    = document.getElementById('turnsBadge');
const ragToggle= document.getElementById('ragToggle');
let turns = 0;

// 自动撑高输入框
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + 'px';
});

inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

function removeWelcome() {
  const w = document.getElementById('welcome');
  if (w) w.remove();
}

function addBubble(role, text='') {
  removeWelcome();
  const row = document.createElement('div');
  row.className = `msg-row ${role}`;

  const av = document.createElement('div');
  av.className = 'avatar';
  av.textContent = role === 'user' ? '我' : 'AI';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  row.appendChild(av);
  row.appendChild(bubble);
  chatEl.appendChild(row);
  chatEl.scrollTop = chatEl.scrollHeight;
  return bubble;
}

async function sendMessage() {
  const query = inputEl.value.trim();
  if (!query) return;

  sendBtn.disabled = true;
  inputEl.value = '';
  inputEl.style.height = 'auto';

  addBubble('user', query);
  const aiBubble = addBubble('ai');
  aiBubble.classList.add('cursor');

  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, use_rag: ragToggle.checked })
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (raw === '[DONE]') break;
        try {
          const obj = JSON.parse(raw);
          if (obj.token)  aiBubble.textContent += obj.token;
          if (obj.turns !== undefined) {
            turns = obj.turns;
            badge.textContent = `${turns} / 5 轮`;
          }
          if (obj.error)  aiBubble.textContent = '⚠ ' + obj.error;
          chatEl.scrollTop = chatEl.scrollHeight;
        } catch {}
      }
    }
  } catch (err) {
    aiBubble.textContent = '⚠ 连接错误：' + err.message;
  } finally {
    aiBubble.classList.remove('cursor');
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

async function clearHistory() {
  await fetch('/clear', { method: 'POST' });
  turns = 0;
  badge.textContent = '0 / 5 轮';
  chatEl.innerHTML = '';
  const w = document.createElement('div');
  w.id = 'welcome';
  w.innerHTML = '<strong>对话已清空</strong>开始新的问答吧';
  chatEl.appendChild(w);
}
</script>
</body>
</html>"""

# ────────────────────────────────────────────────
# 路由
# ────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    query: str = data.get('query', '').strip()
    if not query:
        return jsonify({'error': '请输入问题'}), 400

    def generate():

        # ── 生成历史摘要（供 meta 问题 & system prompt 使用）──
        history_list = dialogue_store.get_history()
        if history_list:
            history_summary = '\n'.join(
                f"第{i+1}轮 用户问：{turn['user'][:80]}"   # 只截取原始 query 部分
                for i, turn in enumerate(history_list)
            )
        else:
            history_summary = '（暂无历史对话）'

        # 知识问答：走 RAG，但 system 层面告知模型可以参考对话历史
        try:
            knowledge_list = ''.join(rrf_fusion(query))
            # print(knowledge_list)
        except Exception as e:
            knowledge_list = f'[检索失败: {e}]'
            # print(knowledge_list)
            # print('~~'*50)
        prompt = (
            f"你可以根据以下【参考内容】和【历史记录】来回答问题。"
            f"如果【参考内容】中没有明确信息，可以使用外部知识，但是必须回答前必须增加提示：根据现有知识库无法知识回答，以下回答来自外模型总结。\n\n"
            f"【参考内容】\n{knowledge_list}\n\n"
            f"【历史内容】\n{history_summary}\n\n"
            f"【问题】\n{query}\n\n"
            f"【回答】"
        )

        # ── 构建消息时，历史里存的是原始 query（非 RAG prompt），保证上下文干净 ──
        # build_messages 会把 history 里的 user/assistant 拼进去
        messages = dialogue_store.build_messages(prompt)

        # ── 流式调用 ──
        full_response = ''
        try:
            response = dialogue_store.client.chat.completions.create(
                model='qwen35_35B',
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                stream=True,
            )
            for chunk in response:
                word = chunk.choices[0].delta.content
                if word:
                    full_response += word
                    payload = json.dumps({'token': word}, ensure_ascii=False)
                    yield f'data: {payload}\n\n'

        except Exception as e:
            err_payload = json.dumps({'error': str(e)}, ensure_ascii=False)
            yield f'data: {err_payload}\n\n'
            return

        # ── 保存历史：user 侧存原始 query（不存 RAG prompt），保持历史可读 ──
        dialogue_store.add_history(query, full_response)
        turns = len(dialogue_store.history)
        yield f'data: {json.dumps({"turns": turns})}\n\n'
        yield 'data: [DONE]\n\n'

    return Response(generate(), mimetype='text/event-stream',
                    headers={'X-Accel-Buffering': 'no',
                             'Cache-Control': 'no-cache'})


@app.route('/clear', methods=['POST'])
def clear():
    dialogue_store.clear()
    return jsonify({'status': 'ok', 'turns': 0})


@app.route('/history', methods=['GET'])
def history():
    return jsonify({'history': dialogue_store.get_history(),
                    'turns': len(dialogue_store.history)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8111, debug=True, threaded=True)