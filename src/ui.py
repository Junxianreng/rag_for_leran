# 启动命令：streamlit run "D:\python_work\master\1\frontend\ui_2.py"
import streamlit as st
import requests
import json
from typing import Generator  # 新增：用于流式生成器

# 后端 API 地址
API_URL = "http://localhost:8000"

# ========== 1. 页面配置 + 自动滚动 JS 逻辑 ==========
st.set_page_config(
    page_title="RAG 分离版客户端",
    page_icon="🤖",
    layout="wide"
)

# 新增：自动滚动到最新消息的 JS 代码
st.components.v1.html(
    """
    <script>
    function autoScroll() {
        const chatContainer = document.querySelector('[data-testid="stChatMessage"]:last-of-type');
        if (chatContainer) {
            chatContainer.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
    }
    setInterval(autoScroll, 100);
    </script>
    """,
    height=0,
    width=0
)

st.title("🤖 zhipu RAG (Client)")

# ========== 2. 侧边栏（保留不变） ==========
with st.sidebar:
    st.header("⚙️ 配置与管理")
    st.subheader("模型选择")
    selected_model = st.selectbox(
        "选择问答模型",
        options=["glm-4.5-air", "glm-3-turbo"],
        index=0,
        help="切换不同的智谱模型进行问答"
    )
    st.divider()
    st.subheader("文档上传")
    uploaded_file = st.file_uploader(
        "上传文档（PDF/TXT/DOCX）",
        type=['pdf', 'txt', 'docx'],
        help="支持PDF、TXT、Word文档格式"
    )
    if uploaded_file and st.button("📤 上传并索引", use_container_width=True):
        with st.spinner("正在发送至后端处理..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(f"{API_URL}/api/upload", files=files)
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ 成功！切分片段数: {data['chunks_added']}")
                    st.rerun()
                else:
                    st.error(f"❌ 失败: {response.text}")
            except Exception as e:
                st.error(f"❌ 连接后端失败: {str(e)}")
    st.divider()
    st.subheader("已上传文档")
    try:
        doc_list_resp = requests.get(f"{API_URL}/api/list_documents")
        if doc_list_resp.status_code == 200:
            docs = doc_list_resp.json()
            if docs:
                for doc in docs:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.text(f"📄 {doc['file_name']}")
                    with col2:
                        if st.button("🗑️", key=f"del_{doc['doc_id']}"):
                            del_resp = requests.post(
                                f"{API_URL}/api/delete_document",
                                json={"doc_id": doc['doc_id']}
                            )
                            if del_resp.status_code == 200:
                                st.success("删除成功！")
                                st.rerun()
                            else:
                                st.error("删除失败！")
            else:
                st.info("暂无已上传文档")
        else:
            st.error("获取文档列表失败")
    except Exception as e:
        st.error(f"加载文档列表出错: {str(e)}")

# ========== 3. 主界面：流式聊天（核心修正：移除nonlocal） ==========
st.subheader("💬 智能问答")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
# 初始化思考过程占位符的会话状态（解决作用域问题）
if "thinking_placeholder" not in st.session_state:
    st.session_state.thinking_placeholder = None

# 展示历史聊天记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📚 参考来源"):
                for src in msg["sources"]:
                    st.text(src)

# 聊天输入框（核心修正：移除nonlocal，改用会话状态存储占位符）
if prompt := st.chat_input("请输入你的问题..."):
    # 1. 展示用户问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 调用后端流式问答接口
    with st.chat_message("assistant"):
        # 思考过程提示（存入会话状态，解决作用域问题）
        st.session_state.thinking_placeholder = st.empty()
        st.session_state.thinking_placeholder.markdown("正在检索知识库中的相关内容...")


        # 定义流式生成器（移除nonlocal，直接使用会话状态的占位符）
        def answer_generator():
            sources = []
            full_response = ""
            try:
                # 流式请求后端
                response = requests.post(
                    f"{API_URL}/api/chat_stream",
                    json={"question": prompt, "model_name": selected_model},
                    stream=True,
                    timeout=30  # 增加超时，避免卡死
                )
                if response.status_code != 200:
                    yield f"❌ 后端报错: {response.text}"
                    return

                # 逐段处理后端响应（无sleep，非阻塞）
                st.session_state.thinking_placeholder.markdown(f"检索完成，正在调用「{selected_model}」模型生成回答...")
                first_chunk = True
                for chunk in response.iter_lines():
                    if chunk:
                        # 收到第一个片段，清空思考提示
                        if first_chunk:
                            st.session_state.thinking_placeholder.empty()
                            first_chunk = False

                        # 解析片段
                        chunk_data = json.loads(chunk.decode('utf-8'))
                        if "sources" in chunk_data:
                            sources = chunk_data["sources"]
                        else:
                            # 逐token返回，无阻塞
                            token = chunk_data.get("content", "")
                            full_response += token
                            yield token  # 生成器返回，由st.write_stream处理

                # 保存最终结果到会话状态
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })

                # 展示参考来源（生成完成后）
                if sources:
                    yield "\n"  # 分隔符
                    # 注意：expander不能在生成器里yield，需提前处理
                    st.session_state["current_sources"] = sources
            except requests.exceptions.ConnectionError:
                yield "❌ 无法连接到后端服务，请确认 backend/main.py 是否在运行！"
            except Exception as e:
                yield f"❌ 问答出错: {str(e)}"
            finally:
                # 确保思考提示被清空
                if st.session_state.thinking_placeholder:
                    st.session_state.thinking_placeholder.empty()


        # 关键：使用st.write_stream（Streamlit原生高效流式API，无卡顿）
        st.write_stream(answer_generator())

        # 展示参考来源（生成完成后单独处理）
        if "current_sources" in st.session_state and st.session_state["current_sources"]:
            with st.expander("📚 参考来源"):
                for src in st.session_state["current_sources"]:
                    st.text(src)
            # 清空临时存储的来源，避免下次复用
            del st.session_state["current_sources"]
