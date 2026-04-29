# -*- coding: utf-8 -*-
"""
课程问答助手 Gradio 前端（集成 RAG 向量检索 + 意图识别）
核心功能：
1. 课程问答核心界面（基于 RAG 向量检索）
2. 意图识别与路由（课程问答/文件操作/闲聊/专业咨询）
3. 课程知识库管理（文档上传/向量库构建/清理）
4. 多用户/多会话隔离
5. 工具调用统计 + 会话管理
6. 健康检查
"""

import gradio as gr
import os
import requests
import json
import time
from datetime import datetime
import logging
import shutil
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
# 添加全局变量记录上次提交时间
last_submit_time = 0
# 尝试导入推荐模块，如果失败则使用 Mock
try:
    from component.recommender import recommender
except ImportError:
    logger.warning("未找到 component.recommender，使用 Mock 替代")
    class MockRecommender:
        def predict(self, thread_id):
            return ["推荐问题 1", "推荐问题 2", "推荐问题 3", "推荐问题 4", "推荐问题 5"]
    recommender = MockRecommender()

# 配置日志
from component.logger import LoggerManager
from component.config import Config
from component.models import ResponseFormat  # 添加这行导入
logger = LoggerManager.get_logger(__name__)

# 导入数据库模块 (增加容错处理)
try:
    from component.memory_sqlite import (
        init_chat_db, get_sessions, create_session, save_message, 
        get_chat_history, create_default_session, delete_session,
        get_context_window, get_relevant_context_for_query
    )
    logger.info("成功导入 memory_sqlite 模块")
except ImportError as e:
    logger.error(f"导入 memory_sqlite 失败: {e}", exc_info=True)
    # 定义 Mock 类以防止程序崩溃
    class MockMemory:
        def init_chat_db(self): pass
        def get_sessions(self): return []
        def create_session(self, *args): return True
        def save_message(self, *args): return True
        def get_chat_history(self, *args): return []
        def delete_session(self, *args): return True
        def get_context_window(self, *args): return []
        def get_relevant_context_for_query(self, *args): return ""
    # 使用 Mock 替代
    init_chat_db = MockMemory().init_chat_db
    get_sessions = MockMemory().get_sessions
    create_session = MockMemory().create_session
    save_message = MockMemory().save_message
    get_chat_history = MockMemory().get_chat_history
    delete_session = MockMemory().delete_session
    get_context_window = MockMemory().get_context_window
    get_relevant_context_for_query = MockMemory().get_relevant_context_for_query

# 导入会话管理模块 (增加容错处理)
try:
    from component.session_manager import SessionManager
    logger.info("成功导入 session_manager 模块")
except ImportError as e:
    logger.error(f"导入 session_manager 失败: {e}", exc_info=True)
    class MockSessionManager:
        def __init__(self, *args): 
            self.current_thread_id = "default_thread"
        def get_current_thread_id(self): return self.current_thread_id
        def switch_session(self, *args): return [], ""
        def new_session(self): return [], ""
        def delete_session(self, *args): return [], ""
        def _update_session_list(self): return ""
        def get_current_session_name(self): return "默认会话"
    SessionManager = MockSessionManager

# 导入文件处理模块 
from component.files_parser import (
    upload_file_stream_to_minio, 
    download_file_from_minio,
    get_file_type_icon,      # 导入图标获取函数
    get_file_type_color,     # 导入颜色获取函数
    generate_file_card_html, # 导入文件卡片HTML生成函数
    generate_file_status_card  # 导入状态卡片HTML生成函数
)

# ==================== 全局配置读取 ====================

API_BASE_URL = Config.API_BASE_URL

# ========== 初始化会话管理器 ==========

try:
    session_manager = SessionManager(
        get_sessions=get_sessions,
        create_session=create_session,
        get_chat_history=get_chat_history,
        delete_session=delete_session
    )
    logger.info("会话管理器初始化成功")
except Exception as e:
    logger.error(f"初始化会话管理器失败: {e}", exc_info=True)
    raise

# ========== 核心 API 调用逻辑 ==========

def chat_response(message: str, history: List[Dict[str, Any]], use_rag: bool = True) -> str:
    """调用 FastAPI 接口获取真实的课程问答回复"""
    current_thread_id = session_manager.get_current_thread_id()
    
    try:
        # 构建请求参数
        payload = {
            "query": message,
            "thread_id": current_thread_id,
            "user_id": "gradio_user",
            "use_rag": use_rag,
            "verbose": True
        }
        
        logger.info(f"发送聊天请求: {message}")
        
        # 调用 FastAPI 核心问答接口
        response = requests.post(
            f"{API_BASE_URL}/api/chat",
            json=payload,
            timeout=600
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["data"]["answer"]
            
            # 保存消息到数据库
            save_message(current_thread_id, "user", message)
            save_message(current_thread_id, "assistant", answer)
            
            logger.info(f"收到回复: {answer[:100]}...")
            return answer
        else:
            error_msg = f"接口调用失败，状态码：{response.status_code}，错误信息：{response.text}"
            logger.error(error_msg)
            return f"❌ {error_msg}"
    
    except requests.exceptions.ConnectionError:
        error_msg = f"无法连接到课程问答服务，请先启动 FastAPI 后端（{API_BASE_URL}）"
        logger.error(error_msg)
        return f"❌ {error_msg}"
    except Exception as e:
        error_msg = f"获取回复失败：{str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"❌ {error_msg}"

# 定义退出函数
def exit_app() -> str:
    """
    退出应用。
    注意：这里我们返回一个空的 HTML，因为实际的关闭逻辑在 JavaScript 的 handleExit 函数中。
    """
    logger.info("用户请求退出")
    return gr.HTML("")

def display_response(response: ResponseFormat, verbose: bool = False):
    """
    终端友好显示回答
    增强RAG上下文显示
    """
    # 使用 Config 中的颜色代码
    print(f"\n{Config.ANSI_AGENT}课程助手回复:{Config.ANSI_END} {response.answer}")

    if verbose:
        if hasattr(response, 'tool_used') and response.tool_used:
            print(f"{Config.ANSI_INFO}使用工具:{Config.ANSI_END} {response.tool_used}")
        if hasattr(response, 'course_references') and response.course_references:
            print(f"{Config.ANSI_KNOWLEDGE}知识点来源:{Config.ANSI_END} {response.course_references}")
        if hasattr(response, 'search_results') and response.search_results:
            print(f"{Config.ANSI_INFO}检索结果:{Config.ANSI_END} {response.search_results}")
        if hasattr(response, 'sql_results') and response.sql_results:
            print(f"{Config.ANSI_INFO}数据结果:{Config.ANSI_END} {response.sql_results}")
        if hasattr(response, 'confidence') and response.confidence is not None:
            print(f"{Config.ANSI_INFO}置信度:{Config.ANSI_END} {response.confidence:.2f}")
    
    # 添加RAG上下文显示
    if hasattr(response, 'rag_context') and response.rag_context:
        print(f"\n{Config.ANSI_KNOWLEDGE}RAG检索上下文:{Config.ANSI_END}")
        print(response.rag_context)
        print("-" * 60)

    print("-" * 60)

# ========== 提交逻辑  ==========

def submit_msg(message: str, history: List[Dict[str, Any]], file_obj: Optional[dict] = None) -> Tuple[str, List[Dict[str, Any]], Optional[dict]]:
    """
    提交消息，支持文本和文件。
    
    1. 接收完整的 history 列表。
    2. 返回更新后的完整 history 列表。
    3. 确保 file_obj 被正确处理。
    4. 返回三个值以匹配输出组件：清空的消息框、更新后的聊天历史、清空的上传按钮
    """
    global last_submit_time  # 声明使用全局变量
    # 防抖机制：如果距离上次提交时间小于1秒，则忽略本次提交
    current_time = time.time()
    if current_time - last_submit_time < 1.0:
        return "", history, None
    
    last_submit_time = current_time
    try:
        # 1. 处理文件上传 (如果有)
        if file_obj is not None:
            # 调用文件处理函数，该函数会更新 history 并返回
            history = handle_file_selection(file_obj, history, session_manager, save_message)
            # 文件上传后，通常不立即发送文字消息，除非用户在输入框也输入了文字
            if not message or not message.strip():
                return "", history, None  # 返回三个值

        # 2. 处理文本消息
        if message and message.strip():
            # 添加用户消息
            history.append({"role": "user", "content": message})
            
            # 调用 API 获取回复
            bot_msg = chat_response(message, history)
            history.append({"role": "assistant", "content": bot_msg})
            
            # 保存消息到数据库
            current_thread_id = session_manager.get_current_thread_id()
            save_message(current_thread_id, "user", message)
            save_message(current_thread_id, "assistant", bot_msg)
        
        # 3. 清空输入框和文件组件
        return "", history, None  # 返回三个值，第三个值用于清空上传按钮
            
    except Exception as e:
        error_msg = f"提交消息失败: {str(e)}"
        logger.error(error_msg, exc_info=True)
        history.append({"role": "assistant", "content": error_msg})
        return "", history, None  # 返回三个值

# ========== 会话相关函数 ==========

def switch_session_wrapper(session_name: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    切换会话的包装函数，用于Gradio事件绑定
    """
    chat_history, session_list_html = session_manager.switch_session(session_name)
    return chat_history, session_list_html

def new_session_wrapper() -> Tuple[List[Dict[str, Any]], str]:
    """
    新建会话的包装函数，用于Gradio事件绑定
    """
    chat_history, session_list_html = session_manager.new_session()
    return chat_history, session_list_html

def delete_session_wrapper(session_name: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    删除会话的包装函数，用于Gradio事件绑定
    """
    chat_history, session_list_html = session_manager.delete_session(session_name)
    return chat_history, session_list_html

def update_session_list_wrapper() -> str:
    """
    更新会话列表的包装函数，用于Gradio事件绑定
    """
    return session_manager._update_session_list()

# ========== 推荐问题相关函数 ==========

def get_suggestions(thread_id: str) -> List[str]:
    """获取当前会话的推荐问题"""
    try:
        return recommender.predict(thread_id)
    except Exception as e:
        logger.error(f"获取推荐问题失败: {e}")
        return []

def on_suggestion_click(btn_text: str, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """点击推荐问题后，将其作为用户消息发送"""
    if not btn_text or btn_text == "...":
        return history
    
    # 添加用户消息
    history.append({"role": "user", "content": btn_text})
    
    # 调用 chat_response 获取回复
    bot_msg = chat_response(btn_text, history)
    history.append({"role": "assistant", "content": bot_msg})
    
    return history

def update_suggestions_after_chat(history: List[Dict[str, Any]]) -> Tuple[str, str, str, str, str]:
    """对话结束后，根据最新的上下文更新推荐问题"""
    current_thread_id = session_manager.get_current_thread_id()
    suggestions = get_suggestions(current_thread_id)
    
    # 确保返回5个值，不足则填充默认值
    if len(suggestions) < 5:
        suggestions += ["..."] * (5 - len(suggestions))
    
    # 返回前5个推荐，分别对应5个按钮
    return suggestions[0], suggestions[1], suggestions[2], suggestions[3], suggestions[4]
def handle_file_selection(file_obj: dict, chat_history: list, session_manager=None, save_message_func=None) -> list:
    """
    当用户点击上传按钮并选择文件后触发
    
    功能：
    1. 直接调用后端接口将文件上传到 MinIO。
    2. 使用新的图标系统生成文件卡片（不显示 minio_path）。
    3. 在聊天界面显示文件卡片和上传结果。
    4. 保存消息到数据库。
    5. 返回更新后的 chat_history。
    """
    if file_obj is None:
        return chat_history
    
    # 获取当前会话 ID，用于保存消息
    current_thread_id = None
    if session_manager is not None:
        current_thread_id = session_manager.get_current_thread_id()
    
    try:
        file_path = file_obj.name
        file_name = os.path.basename(file_path)
        
        logger.info(f"处理文件上传: {file_name}, 路径: {file_path}")
        
        # 1. 直接调用后端接口将文件上传到 MinIO
        upload_status_msg = ""
        minio_path = ""  # 先定义变量
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (file_name, f)}
                payload_data = {'overwrite': True}
                
                upload_url = f"{Config.API_BASE_URL}/api/knowledge/upload"
                logger.info(f"正在上传文件到后端: {upload_url}")
                
                upload_response = requests.post(
                    upload_url,
                    files=files,
                    data=payload_data,  
                    timeout=60
                )
                
                if upload_response.status_code == 200:
                    result = upload_response.json()
                    data = result.get("data", {})
                    
                    # 解析后端返回的数据
                    minio_path = data.get("minio_path", "")  # 这里获取minio_path
                    relevance_score = data.get("relevance_score", 0.0)
                    reason = data.get("reason", "")
                    suggested_categories = data.get("suggested_categories", [])
                    
                    # 获取文件扩展名
                    file_extension = os.path.splitext(file_name)[1].lower()
                    
                    # 获取文件大小
                    file_size = os.path.getsize(file_path)
                    
                    # 生成文件卡片 HTML（不显示 minio_path）
                    file_card_html = generate_file_card_html(
                        file_name=file_name,
                        file_size=file_size,
                        file_extension=file_extension,
                        minio_path=minio_path  # 传入minio_path，但生成时不显示
                    )
                    
                    # 根据相关性生成不同的提示信息
                    if relevance_score > 0.5:
                        upload_status_msg = f"""✅ 文件上传成功: {file_name}
相关性评分: {relevance_score:.2f}
判断理由: {reason}
向量库正在后台更新中..."""
                        if suggested_categories:
                            upload_status_msg += f"\n 建议课程类别: {', '.join(suggested_categories)}"
                    else:
                        upload_status_msg = f"""文件 {file_name} 与课程相关性较低，未上传到向量库
相关性评分: {relevance_score:.2f}
判断理由: {reason}"""
                        if suggested_categories:
                            upload_status_msg += f"\n 建议课程类别: {', '.join(suggested_categories)}"
                    
                    logger.info(f"文件上传成功: {result}")
                else:
                    error_detail = upload_response.text
                    upload_status_msg = f"❌ 文件上传失败: {error_detail}"
                    logger.error(f"文件上传失败: {error_detail}")
                    
        except Exception as e:
            upload_status_msg = f"❌ 文件上传异常: {str(e)}"
            logger.error(f"上传文件时发生异常: {e}", exc_info=True)

        # 2. 将文件卡片作为用户消息添加到历史记录
        chat_history.append({
            "role": "user", 
            "content": file_card_html, 
            "message_type": "file_card",
            "metadata": {
                "file_path": minio_path, 
                "file_name": file_name,
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                "file_type": os.path.splitext(file_name)[1].lower()
            }
        })
        
        # 3. 添加助手回复（上传结果）
        chat_history.append({
            "role": "assistant",
            "content": upload_status_msg
        })
        
        # 4. 保存消息到数据库（确保持久化）
        try:
            if save_message_func is not None and current_thread_id is not None:
                # 保存文件卡片 (用户消息)
                save_message_func(current_thread_id, "user", file_card_html, "file_card")
                
                # 保存上传状态 (助手消息)
                save_message_func(current_thread_id, "assistant", upload_status_msg)
                
                logger.info(f"文件上传消息已保存到数据库，会话ID: {current_thread_id}")
        except Exception as e:
            logger.error(f"保存消息到数据库失败: {e}", exc_info=True)
            # 即使数据库保存失败，也不影响界面显示，只记录日志
        
        logger.info(f"文件已选择、上传并显示: {file_name}")
        
        # 必须返回更新后的 chat_history
        return chat_history
        
    except Exception as e:
        error_msg = f"❌ 处理文件失败: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # 尝试保存错误消息到数据库
        try:
            if save_message_func is not None and current_thread_id is not None:
                save_message_func(current_thread_id, "assistant", error_msg)
        except:
            pass
            
        chat_history.append({"role": "assistant", "content": error_msg})
        return chat_history

# ========== 界面样式配置 ==========

css_path = Path(Config.STYLE_CSS_PATH)
js_path = Path(Config.SCRIPT_JS_PATH)

# 读取 CSS
css_to_use = ""
if css_path.exists():
    with open(css_path, 'r', encoding='utf-8') as f:
        css_to_use = f.read()
    logger.info("成功读取 CSS 文件内容")
else:
    logger.warning(f"找不到 CSS 文件 {css_path}，使用默认样式")
   
# 读取 JS 文件内容
js_content = ""
if js_path.exists():
    try:
        with open(js_path, 'r', encoding='utf-8') as f:
            js_content = f.read()
        logger.info("成功读取 JS 文件内容")
    except Exception as e:
        logger.error(f"读取 JS 文件出错: {e}", exc_info=True)
else:
    logger.warning(f"找不到 JS 文件 {js_path}")

# 获取最新会话名称
latest_session_name = session_manager.get_current_session_name()
latest_history = []

if latest_session_name:
    try:
        latest_history = session_manager.get_chat_history(latest_session_name)
        if not latest_history:
            latest_history = [{"role": "assistant", "content": "你好！很高兴为你服务😊\n\n请输入你的课程相关问题，我会尽力解答！"}]
    except Exception as e:
        logger.error(f"获取最新会话历史失败: {e}")
        latest_history = [{"role": "assistant", "content": "你好！很高兴为你服务😊\n\n请输入你的课程相关问题，我会尽力解答！"}]
else:
    latest_history = [{"role": "assistant", "content": "你好！很高兴为你服务😊\n\n请输入你的课程相关问题，我会尽力解答！"}]

# 自定义 CSS：直接使用读取到的 CSS 内容
custom_css = css_to_use

# 自定义 JavaScript 内容
custom_head = f"""
<script>
    // 定义全局变量，供 JS 使用
    window.API_BASE_URL = "{API_BASE_URL}";
    window.LATEST_SESSION_NAME = "{latest_session_name if latest_session_name else ""}";
    
    // 加载外部 JS 文件内容
    {js_content}
</script>
"""

# ========== 构建界面 ==========

# 注意：custom_css 必须在 with gr.Blocks 之前定义
with gr.Blocks(css=custom_css, head=custom_head, title="课程咨询助手") as app_chat:
    # 顶部导航栏
    with gr.Row(variant="panel", elem_id="top_nav"):
        gr.Markdown("**欢迎！19143614368**", elem_id="user_info")
        # 退出按钮
        exit_btn_html = gr.HTML("""
            <button id="exit_btn" onclick="handleExit(event)" 
                    style="background-color: #ea4335; color: white; border: none; 
                    padding: 6px 12px; border-radius: 20px; cursor: pointer; 
                    font-size: 14px; display: flex; align-items: center; gap: 8px;">
                <svg t="1773642503332" class="icon" viewBox="0 0 1024 1024" version="1.1" 
                     xmlns="http://www.w3.org/2000/svg" width="16" height="16">
                    <path d="M0 192v640c0 70.7 57.3 128 128 128h352c17.7 0 32-14.3 32-32s-14.3-32-32-32H128c-35.3 0-64-28.7-64-64V192c0-35.3 28.7-64 64-64h352c17.7 0 32-14.3 32-32s-14.3-32-32-32H128C57.3 64 0 121.3 0 192z" fill="#ffffff"></path>
                    <path d="M1013.3 488.3L650.9 160.7c-41.2-37.2-106.9-8-106.9 47.5V339c0 4.4-3.6 8-8 8H224c-17.7 0-32 14.3-32 32v266c0 17.7 14.3 32 32 32h312c4.4 0 8-3.6 8-8v130.9c0 55.5 65.8 84.7 106.9 47.5l362.4-327.6c14.1-12.8 14.1-34.8 0-47.5zM256 597V427c0-8.8 7.2-16 16-16h304c17.7 0 32-14.3 32-32V244.9c0-13.9 16.4-21.2 26.7-11.9L938 506.1c3.5 3.2 3.5 8.7 0 11.9L634.7 791c-10.3 9.3-26.7 2-26.7-11.9V645c0-17.7-14.3-32-32-32H272c-8.8 0-16-7.2-16-16z" fill="#ffffff"></path>
                </svg>
                退出
            </button>
        """)
    
    # 主区域布局
    with gr.Row(equal_height=False):
        # 左侧会话历史栏
        with gr.Column(scale=2, min_width=220, elem_id="left_panel"):
            # 新建会话按钮
            new_chat_btn = gr.Button("+ 新建会话", variant="primary", elem_id="new_chat")
            
            # 聊天记录标题
            gr.Markdown("### 聊天记录", elem_id="chat_history_title", elem_classes="sidebar-header")
            
            # 会话列表HTML
            session_list_html = gr.HTML(value=update_session_list_wrapper(), elem_id="session_list")

            # 隐藏的输入框
            session_name_input = gr.Textbox(
                value="",
                elem_id="session_name_input",
                elem_classes="hidden-component",
                show_label=False,
                visible=True, 
                container=False,
                interactive=True
            )
            
            # 隐藏的触发按钮
            trigger_switch_btn = gr.Button(
                elem_id="trigger_switch_btn", 
                elem_classes="ghost-btn"
            )
            
            # 隐藏的删除按钮
            delete_trigger_btn = gr.Button(
                elem_id="delete_trigger_btn",
                elem_classes="ghost-btn"
            )

        # 右侧聊天界面
        with gr.Column(scale=8, elem_classes="main-content"):
            gr.Markdown("## 欢迎来到课程咨询小助手", elem_id="chat_title")
            
            # 聊天窗口
            # 使用 value 初始化，并确保后续函数返回的是字典列表格式
           
            chatbot = gr.Chatbot(
                value=latest_history, 
                elem_id="chatbot",
                label=None,
                autoscroll=True,
                elem_classes=["custom-chatbot"],
                sanitize_html=False
            )
            
            # === 推荐问题区域 ===
            with gr.Row(elem_id="suggestions_area"):
                suggestion_btn_1 = gr.Button("...", size="sm", elem_classes="suggestion-btn")
                suggestion_btn_2 = gr.Button("...", size="sm", elem_classes="suggestion-btn")
                suggestion_btn_3 = gr.Button("...", size="sm", elem_classes="suggestion-btn")
                suggestion_btn_4 = gr.Button("...", size="sm", elem_classes="suggestion-btn")
                suggestion_btn_5 = gr.Button("...", size="sm", elem_classes="suggestion-btn")
            
            # 底部输入栏
            with gr.Row(elem_id="input_area"):
                # 1. 上传按钮 (使用 Gradio 原生 UploadButton + icon 参数)
                upload_btn = gr.UploadButton(
                    "上传",
                    file_types=[".txt", ".md", ".json", ".csv", ".docx", ".pdf"],
                    elem_id="file_upload_btn_real",
                    scale=0,
                    # 使用 icon 参数
                    icon="assets/upload.svg"
                )
                
                # 2. 输入框
                msg = gr.Textbox(
                    placeholder="输入你的课程问题...",
                    scale=5, 
                    elem_id="msg_input",
                    show_label=False,
                    lines=1,
                    container=False,
                    autofocus=True
                )
                
                # 3. 发送按钮 (使用 Gradio 原生 Button + icon 参数)
                send_btn = gr.Button(
                    "发送",
                    variant="primary",
                    scale=0,
                    elem_id="send_btn_real",
                    # 使用 icon 参数
                    icon="assets/send.svg"
                )

    # ========== 绑定事件 ==========
    
    # 1. 推荐问题点击事件
    for btn in [suggestion_btn_1, suggestion_btn_2, suggestion_btn_3, suggestion_btn_4, suggestion_btn_5]:
        btn.click(
            fn=on_suggestion_click,
            inputs=[btn, chatbot],
            outputs=[chatbot]
        )
    
    # 2. 聊天内容变化时，更新推荐问题
    chatbot.change(
        fn=update_suggestions_after_chat,
        inputs=[chatbot],
        outputs=[suggestion_btn_1, suggestion_btn_2, suggestion_btn_3, suggestion_btn_4, suggestion_btn_5]
    )
    
    # 3. 切换会话
    trigger_switch_btn.click(
        fn=switch_session_wrapper,
        inputs=session_name_input,
        outputs=[chatbot, session_list_html]
    )

    # 4. 删除会话
    delete_trigger_btn.click(
        fn=delete_session_wrapper,
        inputs=session_name_input,
        outputs=[chatbot, session_list_html]
    )
    
    # 5. 新建会话按钮
    new_chat_btn.click(
        fn=new_session_wrapper,
        outputs=[chatbot, session_list_html]
    )
    
    # 6. 文件上传事件 (立即上传并更新向量库)
    upload_btn.upload(
        fn=lambda file_obj, chat_history: handle_file_selection(
            file_obj, 
            chat_history, 
            session_manager,  # 传递 session_manager
            save_message      # 传递 save_message 函数
        ),
        inputs=[upload_btn, chatbot],
        outputs=[chatbot]
    )
    
    # 7. 发送消息（回车键）
    msg.submit(
        fn=submit_msg, 
        inputs=[msg, chatbot, upload_btn], 
        outputs=[msg, chatbot, upload_btn]
    )
    
    # 8. 发送按钮
    send_btn.click(
        fn=submit_msg, 
        inputs=[msg, chatbot, upload_btn], 
        outputs=[msg, chatbot, upload_btn]
    )
    
    # 9. 退出按钮
    exit_btn_html.click(fn=exit_app)

# ========== 启动应用 ==========
if __name__ == "__main__":
    # 启动 Gradio 页面
    try:
        logger.info("启动 Gradio 应用...")
        app_chat.launch(
            server_name="127.0.0.1", 
            server_port=7860,
            share=False,  # 不生成公网链接
            debug=True,   # 调试模式
            show_error=True
        )
    except KeyboardInterrupt:
        logger.info("应用已正常停止")
    except Exception as e:
        logger.error(f"启动应用时出错: {e}", exc_info=True)
