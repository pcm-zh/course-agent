# -*- coding: utf-8 -*-
"""
记忆持久化模块
提供混合存储方案：
1. 内存存储：用于LangGraph Agent的检查点状态
2. 持久化存储：用于用户聊天历史记录
核心功能：
1. 为LangGraph提供线程安全的内存检查点存储
2. 提供聊天历史的SQLite持久化存储（会话/消息）
3. 提供统一的会话管理接口
4. 支持异步测试和数据清理功能
5. 确保各个会话互不干扰
"""
import asyncio
import threading
import os
import sqlite3
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from contextlib import contextmanager
import json

from langgraph.checkpoint.memory import MemorySaver

# 导入项目配置和日志模块
from .config import Config
from .logger import LoggerManager

# 创建当前模块的日志实例
logger = LoggerManager.get_logger(__name__)

# ==================== 全局单例管理 ====================
_saver_instance: Optional[MemorySaver] = None
_instance_lock = threading.Lock()
_thread_local = threading.local()  # 线程本地存储，用于隔离不同会话的数据
_schema_checked = False  # 用于标记是否已经检查过数据库结构

def get_saver() -> MemorySaver:
    """
    获取LangGraph检查点保存器实例（单例模式）
    注意：当前使用 MemorySaver，数据仅保存在内存中，重启后会丢失。
    """
    global _saver_instance
    if _saver_instance is None:
        with _instance_lock:
            if _saver_instance is None:
                try:
                    logger.info("初始化 MemorySaver (内存存储模式)")
                    _saver_instance = MemorySaver()
                    logger.info("MemorySaver 初始化完成")
                except Exception as e:
                    logger.error(f"初始化 MemorySaver 失败: {str(e)}", exc_info=True)
                    raise RuntimeError(f"无法初始化 MemorySaver: {str(e)}") from e
    return _saver_instance

def reset_saver():
    """
    重置MemorySaver实例（用于测试/重新初始化场景）
    """
    global _saver_instance
    with _instance_lock:
        if _saver_instance is not None:
            try:
                logger.info("已重置 MemorySaver 实例")
            except Exception as e:
                logger.error(f"重置 MemorySaver 时出错: {str(e)}", exc_info=True)
            finally:
                _saver_instance = None

def clear_memory():
    """
    清空所有内存检查点数据（危险操作！）
    直接操作 MemorySaver 内部的 storage 字典
    """
    saver = get_saver()
    with _instance_lock:
        if hasattr(saver, 'storage'):
            saver.storage.clear()
            logger.warning("已清空所有内存检查点数据")
        else:
            logger.warning("无法访问 MemorySaver 的 storage 属性，清空操作可能失败")

def get_checkpoint_count(thread_id: str = None) -> int:
    """
    获取内存检查点数量
    直接从 MemorySaver 内部存储统计
    """
    try:
        saver = get_saver()
        if not hasattr(saver, 'storage'):
            return 0
            
        with _instance_lock:
            storage = saver.storage
            if thread_id:
                # MemorySaver 的 key 通常是 tuple: (thread_id, checkpoint_ns)
                return sum(1 for k in storage.keys() if k[0] == thread_id)
            else:
                return len(storage)
    except Exception as e:
        logger.error(f"获取检查点数量时出错: {str(e)}", exc_info=True)
        return 0

def list_threads() -> list:
    """
    列出所有存在内存检查点的线程ID
    """
    try:
        saver = get_saver()
        if not hasattr(saver, 'storage'):
            return []
            
        with _instance_lock:
            threads = set()
            for k in saver.storage.keys():
                if isinstance(k, tuple) and len(k) > 0:
                    threads.add(k[0])
            return list(threads)
    except Exception as e:
        logger.error(f"列出线程时出错: {str(e)}", exc_info=True)
        return []

def clear_thread(thread_id: str):
    """
    清除指定线程的内存检查点
    """
    try:
        saver = get_saver()
        if not hasattr(saver, 'storage'):
            return

        with _instance_lock:
            storage = saver.storage
            keys_to_delete = [k for k in storage.keys() if k[0] == thread_id]
            for k in keys_to_delete:
                del storage[k]
            
            if keys_to_delete:
                logger.info(f"已清除线程 {thread_id} 的 {len(keys_to_delete)} 个检查点")
            else:
                logger.info(f"线程 {thread_id} 不存在，无需清除")
                
    except Exception as e:
        logger.error(f"清除线程 {thread_id} 时出错: {str(e)}", exc_info=True)
        raise RuntimeError(f"无法清除线程 {thread_id}: {str(e)}") from e

async def test_saver():
    """
    异步测试MemorySaver功能
    """
    try:
        saver = get_saver()
        config = {"configurable": {"thread_id": "test_123"}}
        
        # 1. 创建检查点
        checkpoint_id = await saver.aput(
            config=config,
            checkpoint={"data": "test_value"},
            metadata={"step": 1}
        )
        logger.info(f"创建检查点 ID: {checkpoint_id}")
        
        # 2. 读取检查点
        checkpoint = await saver.aget(config)
        logger.info(f"读取检查点内容: {checkpoint}")
        
        # 3. 验证统计功能
        count = get_checkpoint_count("test_123")
        logger.info(f"线程 test_123 的检查点数量: {count}")  # 预期输出 1
        
        # 4. 验证线程列表
        threads = list_threads()
        logger.info(f"当前活跃线程: {threads}")  # 预期包含 'test_123'
        
        # 5. 清理
        clear_thread("test_123")
        final_count = get_checkpoint_count("test_123")
        logger.info(f"清理后线程 test_123 的检查点数量: {final_count}")  # 预期输出 0
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}", exc_info=True)
        raise

# 保持别名兼容性
get_sqlite_saver = get_saver


# ==================== 聊天历史数据库功能 ====================

def _ensure_db_schema(conn: sqlite3.Connection):
    """
    确保数据库表结构是最新的（内部函数）
    如果检测到缺失列，会自动执行 ALTER TABLE 添加
    """
    global _schema_checked
    # 使用全局锁防止多线程并发修改表结构
    with _instance_lock:
        # 如果已经检查过，直接跳过（除非你想每次都检查，可以去掉这个判断）
        if _schema_checked:
            return

        try:
            cursor = conn.cursor()
            
            # 1. 确保表存在
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL UNIQUE,
                create_time TEXT NOT NULL,
                user_id TEXT NOT NULL DEFAULT 'gradio_user',
                title TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}'
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                create_time TEXT NOT NULL,
                message_type TEXT DEFAULT 'text',
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (thread_id) REFERENCES sessions(thread_id)
            )
            """)
            
            # 2. 检查并添加 messages 表缺失的列
            cursor.execute("PRAGMA table_info(messages)")
            columns = [info[1] for info in cursor.fetchall()]
            
            if 'message_type' not in columns:
                logger.info("正在添加 message_type 列...")
                cursor.execute("ALTER TABLE messages ADD COLUMN message_type TEXT DEFAULT 'text'")
                
            if 'metadata' not in columns:
                logger.info("正在添加 metadata 列...")
                cursor.execute("ALTER TABLE messages ADD COLUMN metadata TEXT DEFAULT '{}'")
            
            # 3. 创建索引
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_thread_id 
            ON messages(thread_id)
            """)
            
            conn.commit()
            _schema_checked = True
            logger.debug("数据库结构检查完成")
            
        except Exception as e:
            logger.error(f"检查/更新数据库结构失败: {e}", exc_info=True)
            # 发生错误不抛出异常，避免阻塞连接获取，但记录日志
            # 如果必须严格保证结构，可以在这里 raise

@contextmanager
def get_chat_db_connection():
    """
    获取聊天历史数据库连接的上下文管理器
    每个线程获取独立的连接，确保会话隔离
    并且在首次连接时自动检查并修复数据库结构
    
    Yields:
        sqlite3.Connection: 数据库连接对象
    """
    # 使用线程本地存储来隔离不同线程的数据库连接
    if not hasattr(_thread_local, 'db_connection'):
        _thread_local.db_connection = None
    
    db_path = Config.CHAT_HISTORY_DB_PATH
    
    # 确保数据库目录存在
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"创建数据库目录: {db_dir}")
        except Exception as e:
            logger.error(f"创建数据库目录失败: {db_dir}, 错误: {e}")
            raise

    conn = None
    try:
        # 如果当前线程已有连接且未关闭，则重用
        if _thread_local.db_connection is None:
            conn = sqlite3.connect(Config.CHAT_HISTORY_DB_PATH)
            conn.row_factory = sqlite3.Row  # 允许通过列名访问
            _thread_local.db_connection = conn
        else:
            conn = _thread_local.db_connection
        
        # 在返回连接之前，强制检查并修复表结构
        _ensure_db_schema(conn)
        # ---------------------------------------------------
            
        yield conn
    except Exception as e:
        logger.error(f"数据库操作发生错误: {e}", exc_info=True)
        if conn:
            try:
                conn.rollback()
            except:
                pass
        raise
    finally:
        # 不在这里关闭连接，而是在线程结束时关闭
        pass

def close_thread_db_connection():
    """
    关闭当前线程的数据库连接
    应在线程结束时调用
    """
    if hasattr(_thread_local, 'db_connection') and _thread_local.db_connection is not None:
        try:
            _thread_local.db_connection.close()
            _thread_local.db_connection = None
            logger.debug("已关闭当前线程的数据库连接")
        except Exception as e:
            logger.error(f"关闭数据库连接时出错: {e}", exc_info=True)

def init_chat_db():
    """
    初始化聊天历史数据库（兼容旧接口）
    现在主要由 get_chat_db_connection 中的 _ensure_db_schema 接管
    """
    try:
        with get_chat_db_connection() as conn:
            # 连接建立时会自动调用 _ensure_db_schema
            pass
        return True
    except Exception as e:
        logger.error(f"初始化聊天历史数据库失败: {e}", exc_info=True)
        return False

def _ensure_tables_exist():
    """确保数据库表存在（内部辅助函数，已弃用，保留以防旧代码调用）"""
    # 逻辑已合并到 get_chat_db_connection -> _ensure_db_schema 中
    # 这里直接调用初始化即可
    return init_chat_db()

def get_sessions() -> List[Dict]:
    """获取所有会话"""
    try:
        with get_chat_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT thread_id, create_time, title, metadata FROM sessions ORDER BY create_time DESC")
            sessions = cursor.fetchall()
            
            # 构建会话列表
            session_list = []
            for sess in sessions:
                thread_id = sess["thread_id"]
                create_time_str = sess["create_time"]
                title = sess["title"] or thread_id
                metadata = {}
                
                # 安全解析 metadata
                try:
                    if sess["metadata"]:
                        metadata = json.loads(sess["metadata"])
                except:
                    pass
                
                # 安全解析 create_time 
                try:
                    create_time = datetime.fromisoformat(create_time_str)
                except Exception:
                    # 如果解析失败，使用当前时间，并记录警告
                    logger.warning(f"会话 {thread_id} 的时间格式无效: {create_time_str}，已使用当前时间替代")
                    create_time = datetime.now()
                
                session_list.append({
                    "name": thread_id,
                    "title": title,
                    "create_time": create_time,
                    "metadata": metadata
                })
            
            return session_list
    except Exception as e:
        logger.error(f"获取会话列表时发生数据库错误: {str(e)}")
        return []

def create_session(thread_id: str, title: str = None, metadata: Dict = None) -> bool:
    """创建新会话（如果已存在则跳过）"""
    try:
        with get_chat_db_connection() as conn:
            cursor = conn.cursor()
            title = title or thread_id
            metadata_json = json.dumps(metadata) if metadata else '{}'
            
            # 检查会话是否已存在
            cursor.execute("""
            SELECT thread_id FROM sessions WHERE thread_id = ?
            """, (thread_id,))
            existing = cursor.fetchone()
            
            if existing:
                logger.info(f"会话 {thread_id} 已存在，跳过创建")
                return True
            
            # 插入新会话
            cursor.execute("""
            INSERT INTO sessions (thread_id, create_time, title, metadata)
            VALUES (?, ?, ?, ?)
            """, (thread_id, datetime.now().isoformat(), title, metadata_json))
            
            conn.commit()
            logger.info(f"成功创建会话: {thread_id}")
            return True
            
    except Exception as e:
        logger.error(f"创建会话失败: {e}", exc_info=True)
        return False

def update_session(thread_id: str, title: str = None, metadata: Dict = None) -> bool:
    """更新会话信息"""
    try:
        with get_chat_db_connection() as conn:
            cursor = conn.cursor()
            
            if title is not None and metadata is not None:
                metadata_json = json.dumps(metadata)
                cursor.execute("""
                UPDATE sessions SET title = ?, metadata = ? WHERE thread_id = ?
                """, (title, metadata_json, thread_id))
            elif title is not None:
                cursor.execute("""
                UPDATE sessions SET title = ? WHERE thread_id = ?
                """, (title, thread_id))
            elif metadata is not None:
                metadata_json = json.dumps(metadata)
                cursor.execute("""
                UPDATE sessions SET metadata = ? WHERE thread_id = ?
                """, (metadata_json, thread_id))
            
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"更新会话失败: {e}", exc_info=True)
        return False

def delete_session(thread_id: str) -> bool:
    """删除指定会话及其所有消息"""
    if not thread_id:
        logger.error("错误：thread_id 为空")
        return False

    try:
        with get_chat_db_connection() as conn:
            cursor = conn.cursor()
            
            # 先删除该会话下的所有消息
            cursor.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
            deleted_rows = cursor.rowcount
            logger.info(f"已删除会话 {thread_id} 下的 {deleted_rows} 条消息")
            
            # 再删除会话记录
            cursor.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
            
            conn.commit()
            logger.info(f"成功删除会话: {thread_id}")
            return True
            
    except sqlite3.Error as e:
        logger.error(f"数据库错误: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"未知错误: {e}", exc_info=True)
        return False

def save_message(thread_id: str, role: str, content: str, message_type: str = "text", metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    保存消息到数据库
    
    Args:
        thread_id: 会话ID
        role: 消息角色 (user/assistant)
        content: 消息内容
        message_type: 消息类型 (text/file_card)
        metadata: 可选的元数据字典
    
    Returns:
        保存是否成功
    """
    try:
        # 使用统一的连接管理器
        with get_chat_db_connection() as conn:
            cursor = conn.cursor()
            
            # 将元数据转换为JSON字符串
            metadata_json = json.dumps(metadata) if metadata else None
            
            # 插入消息记录，注意使用 create_time 而不是 timestamp
            cursor.execute(
                """
                INSERT INTO messages (thread_id, role, content, message_type, metadata, create_time)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (thread_id, role, content, message_type, metadata_json, datetime.now().isoformat())
            )
            
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"保存消息失败: {e}", exc_info=True)
        return False

def get_chat_history(thread_id: str) -> List[Dict[str, Any]]:
    """
    从数据库获取聊天历史
    
    Args:
        thread_id: 会话ID
    
    Returns:
        聊天历史列表
    """
    try:
        # 使用统一的连接管理器
        with get_chat_db_connection() as conn:
            conn.row_factory = sqlite3.Row  # 允许通过列名访问
            cursor = conn.cursor()
            
            # 查询消息记录，使用 create_time 排序
            cursor.execute(
                """
                SELECT role, content, message_type, metadata
                FROM messages
                WHERE thread_id = ?
                ORDER BY create_time ASC
                """,
                (thread_id,)
            )
            
            rows = cursor.fetchall()
            
            # 构建聊天历史
            history = []
            for row in rows:
                message = {
                    "role": row["role"],
                    "content": row["content"],
                    "message_type": row["message_type"]
                }
                
                # 如果有元数据，解析并添加到消息中
                if row["metadata"]:
                    try:
                        message["metadata"] = json.loads(row["metadata"])
                    except:
                        pass
                
                history.append(message)
            
            return history
    except Exception as e:
        logger.error(f"获取聊天历史失败: {e}", exc_info=True)
        return []

def get_session(thread_id: str) -> Optional[Dict]:
    """获取单个会话信息"""
    try:
        with get_chat_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT thread_id, create_time, title, metadata FROM sessions 
            WHERE thread_id = ?
            """, (thread_id,))
            
            sess = cursor.fetchone()
            if not sess:
                return None
                
            create_time_str = sess["create_time"]
            title = sess["title"] or sess["thread_id"]
            metadata = json.loads(sess["metadata"]) if sess["metadata"] else {}
            
            try:
                create_time = datetime.fromisoformat(create_time_str)
            except:
                create_time = datetime.now()
            
            return {
                "name": sess["thread_id"],
                "title": title,
                "create_time": create_time,
                "metadata": metadata
            }
    except Exception as e:
        logger.error(f"获取会话失败: {e}", exc_info=True)
        return None

def create_default_session() -> bool:
    """创建默认会话"""
    thread_id = "默认会话"
    try:
        with get_chat_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR IGNORE INTO sessions (thread_id, create_time, title)
            VALUES (?, ?, ?)
            """, (thread_id, datetime.now().isoformat(), "默认会话"))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"创建默认会话失败: {e}", exc_info=True)
        return False

def search_messages(keyword: str, thread_id: str = None) -> List[Dict]:
    """搜索包含关键词的消息"""
    try:
        with get_chat_db_connection() as conn:
            cursor = conn.cursor()
            
            if thread_id:
                cursor.execute("""
                SELECT thread_id, role, content, create_time FROM messages 
                WHERE thread_id = ? AND content LIKE ?
                ORDER BY create_time DESC
                """, (thread_id, f"%{keyword}%"))
            else:
                cursor.execute("""
                SELECT thread_id, role, content, create_time FROM messages 
                WHERE content LIKE ?
                ORDER BY create_time DESC
                """, (f"%{keyword}%",))
            
            messages = cursor.fetchall()
            return [
                {
                    "thread_id": msg["thread_id"],
                    "role": msg["role"],
                    "content": msg["content"],
                    "create_time": msg["create_time"]
                } for msg in messages
            ]
    except Exception as e:
        logger.error(f"搜索消息失败: {e}", exc_info=True)
        return []

def get_session_message_count(thread_id: str) -> int:
    """获取指定会话的消息数量"""
    try:
        with get_chat_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT COUNT(*) as count FROM messages WHERE thread_id = ?
            """, (thread_id,))
            
            result = cursor.fetchone()
            return result["count"] if result else 0
    except Exception as e:
        logger.error(f"获取消息数量失败: {e}", exc_info=True)
        return 0

def get_all_message_counts() -> Dict[str, int]:
    """获取所有会话的消息数量"""
    try:
        with get_chat_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT thread_id, COUNT(*) as count FROM messages 
            GROUP BY thread_id
            """)
            
            results = cursor.fetchall()
            return {row["thread_id"]: row["count"] for row in results}
    except Exception as e:
        logger.error(f"获取所有消息数量失败: {e}", exc_info=True)
        return {}

def get_recent_sessions(limit: int = 10) -> List[Dict]:
    """获取最近的会话"""
    try:
        with get_chat_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT thread_id, create_time, title, metadata FROM sessions 
            ORDER BY create_time DESC
            LIMIT ?
            """, (limit,))
            
            sessions = cursor.fetchall()
            session_list = []
            
            for sess in sessions:
                thread_id = sess["thread_id"]
                create_time_str = sess["create_time"]
                title = sess["title"] or thread_id
                metadata = json.loads(sess["metadata"]) if sess["metadata"] else {}
                
                try:
                    create_time = datetime.fromisoformat(create_time_str)
                except:
                    create_time = datetime.now()
                
                session_list.append({
                    "name": thread_id,
                    "title": title,
                    "create_time": create_time,
                    "metadata": metadata
                })
            
            return session_list
    except Exception as e:
        logger.error(f"获取最近会话失败: {e}", exc_info=True)
        return []

def get_active_sessions(days: int = 7) -> List[Dict]:
    """获取最近几天内有活动的会话"""
    try:
        with get_chat_db_connection() as conn:
            cursor = conn.cursor()
            
            # 计算日期阈值
            threshold_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
            SELECT DISTINCT s.thread_id, s.create_time, s.title, s.metadata 
            FROM sessions s
            JOIN messages m ON s.thread_id = m.thread_id
            WHERE m.create_time >= ?
            ORDER BY m.create_time DESC
            """, (threshold_date,))
            
            sessions = cursor.fetchall()
            session_list = []
            
            for sess in sessions:
                thread_id = sess["thread_id"]
                create_time_str = sess["create_time"]
                title = sess["title"] or thread_id
                metadata = json.loads(sess["metadata"]) if sess["metadata"] else {}
                
                try:
                    create_time = datetime.fromisoformat(create_time_str)
                except:
                    create_time = datetime.now()
                
                session_list.append({
                    "name": thread_id,
                    "title": title,
                    "create_time": create_time,
                    "metadata": metadata
                })
            
            return session_list
    except Exception as e:
        logger.error(f"获取活跃会话失败: {e}", exc_info=True)
        return []

def export_session(thread_id: str) -> Optional[Dict]:
    """导出会话及其所有消息"""
    try:
        session = get_session(thread_id)
        if not session:
            return None
            
        messages = get_chat_history(thread_id)
        
        return {
            "session": session,
            "messages": messages
        }
    except Exception as e:
        logger.error(f"导出会话失败: {e}", exc_info=True)
        return None

def import_session(session_data: Dict) -> bool:
    """导入会话及其所有消息"""
    try:
        session = session_data.get("session")
        messages = session_data.get("messages", [])
        
        if not session:
            return False
            
        thread_id = session["name"]
        title = session.get("title", thread_id)
        metadata = session.get("metadata", {})
        
        # 创建会话
        if not create_session(thread_id, title, metadata):
            return False
            
        # 导入消息
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            msg_type = msg.get("message_type", "text")  # 获取类型，默认为text
            if role and content:
                save_message(thread_id, role, content, msg_type)
                
        return True
    except Exception as e:
        logger.error(f"导入会话失败: {e}", exc_info=True)
        return False

def repair_database():
    """
    修复数据库中的脏数据（主要是时间格式）
    尝试将非标准的时间字符串转换为标准格式
    """
    conn = None
    try:
        db_path = Config.CHAT_HISTORY_DB_PATH
        if not os.path.exists(db_path):
            logger.warning("数据库文件不存在，无需修复")
            return False
        
        conn = sqlite3.connect(Config.CHAT_HISTORY_DB_PATH)
        cursor = conn.cursor()
        
        # 1. 查找所有时间格式不标准的记录（例如包含空格而不是T）
        # 这里假设标准格式是 ISO 8601 (YYYY-MM-DDTHH:MM:SS)
        # 脏数据可能是 YYYY-MM-DD HH:MM:SS
        cursor.execute("SELECT thread_id, create_time FROM sessions")
        rows = cursor.fetchall()
        
        updated_count = 0
        for row in rows:
            thread_id = row["thread_id"]
            old_time = row["create_time"]
            
            # 如果包含空格，尝试替换为 T
            if ' ' in old_time and 'T' not in old_time:
                try:
                    # 尝试解析以确认它是有效的日期时间
                    # 这里简单处理：直接替换空格为 T
                    new_time = old_time.replace(' ', 'T')
                    
                    # 尝试转换回 datetime 对象以验证格式
                    datetime.fromisoformat(new_time)
                    
                    # 更新数据库
                    cursor.execute("UPDATE sessions SET create_time = ? WHERE thread_id = ?", (new_time, thread_id))
                    updated_count += 1
                    logger.info(f"修复会话 {thread_id} 的时间: {old_time} -> {new_time}")
                except Exception as e:
                    logger.warning(f"无法修复会话 {thread_id} 的时间 {old_time}: {e}")
        
        if updated_count > 0:
            conn.commit()
            logger.info(f"数据库修复完成，共更新 {updated_count} 条记录")
            return True
        else:
            logger.info("数据库未发现需要修复的脏数据")
            return True
            
    except Exception as e:
        logger.error(f"修复数据库失败: {e}", exc_info=True)
        return False
    finally:
        if conn:
            conn.close()

def get_context_window(thread_id: str, max_turns: int = 5, max_tokens: int = 3000) -> List[Dict[str, str]]:
    """
    获取指定会话的上下文窗口（用于 LLM Prompt）
    
    Args:
        thread_id: 会话ID
        max_turns: 最大保留轮数
        max_tokens: 最大保留 Token 数（估算值，中文字符*1.5）
    
    Returns:
        格式化后的消息列表，按时间倒序排列（最新的在前）
        [{"role": "user", "content": "..."}, ...]
    """
    try:
        with get_chat_db_connection() as conn:
            cursor = conn.cursor()
            
            # 1. 获取最近 N 条消息（倒序）
            cursor.execute(
                "SELECT role, content, message_type FROM messages WHERE thread_id = ? ORDER BY create_time DESC LIMIT ?",
                (thread_id, max_turns * 2)
            )
            rows = cursor.fetchall()
            
            context = []
            current_tokens = 0
            
            # 2. 遍历并截断（基于 Token 估算）
            for row in rows:
                content = row['content']
                # 简单估算 Token 数：中文约 1.5 字符/token，英文约 0.3 词/token
                # 这里简化为 字符数 * 1.5
                estimated_tokens = len(content) * 1.5
                
                if current_tokens + estimated_tokens > max_tokens:
                    logger.info(f"上下文窗口已满 ({current_tokens}/{max_tokens} tokens)，截断历史记录")
                    break
                
                context.append({
                    "role": row['role'],
                    "content": content,
                    "message_type": row['message_type'] if 'message_type' in row.keys() else "text"
                })
                current_tokens += estimated_tokens
            
            # 3. 再次倒序，使其符合 Chat API 的时间顺序要求（旧 -> 新）
            context.reverse()
            
            return context
            
    except Exception as e:
        logger.error(f"获取上下文窗口失败: {e}")
        return []

def get_thread_summary(thread_id: str) -> str:
    """
    获取会话的简要摘要，由最后一条用户消息和最后一条助手回复组成
    """
    try:
        with get_chat_db_connection() as conn:
            cursor = conn.cursor()
            
            # 获取最后一条用户消息
            cursor.execute(
                "SELECT content FROM messages WHERE thread_id = ? AND role = 'user' ORDER BY create_time DESC LIMIT 1",
                (thread_id,)
            )
            user_msg = cursor.fetchone()
            
            # 获取最后一条助手回复
            cursor.execute(
                "SELECT content FROM messages WHERE thread_id = ? AND role = 'assistant' ORDER BY create_time DESC LIMIT 1",
                (thread_id,)
            )
            assistant_msg = cursor.fetchone()
            
            summary_parts = []
            if user_msg:
                summary_parts.append(f"用户: {user_msg['content'][:50]}...")
            if assistant_msg:
                summary_parts.append(f"助手: {assistant_msg['content'][:50]}...")
                
            return " | ".join(summary_parts) if summary_parts else "新会话"
            
    except Exception as e:
        logger.error(f"获取会话摘要失败: {e}")
        return "未知会话"


def get_relevant_context_for_query(thread_id: str, query: str, rag_course=None) -> str:
    """
    获取与用户查询相关的上下文（用于 RAG）
    
    核心逻辑：
    1. 获取当前会话的聊天历史。
    2. 检查历史中是否存在刚刚上传的文件（通过 message_type == "file_card" 判断）。
    3. 如果存在，就强制检索该文件的内容（使用元数据过滤）。
    4. 将检索到的文档内容与聊天历史合并，构建成一个完整的上下文字符串。
    
    Args:
        thread_id: 会话ID
        query: 用户查询
        rag_course: 向量存储实例（可选，如果为None则尝试获取）
    
    Returns:
        完整的上下文字符串（聊天历史 + 检索到的文档内容）
    """
    try:
        # 1. 获取当前会话的聊天历史
        history = get_chat_history(thread_id)
        
        # 2. 检查历史中是否存在刚刚上传的文件
        uploaded_file_path = None
        uploaded_file_name = None
        
        # 倒序遍历历史记录，找到最近的一个文件卡片
        for item in reversed(history):
            if item.get("role") == "user" and item.get("message_type") == "file_card":
                # 从元数据中获取文件路径和名称
                metadata = item.get("metadata", {})
                uploaded_file_path = metadata.get("minio_path")
                uploaded_file_name = metadata.get("file_name")
                
                # 如果有 MinIO 路径，转换为本地路径格式用于检索
                if uploaded_file_path and uploaded_file_path.startswith("minio://"):
                    # 从 MinIO 路径中提取对象名称
                    # 格式: minio://bucket/course_materials/object_name
                    parts = uploaded_file_path.split("/")
                    if len(parts) >= 4:
                        object_name = parts[-1]
                        # 构建本地文件路径（用于向量库检索）
                        # 注意：这里需要根据实际的向量库存储路径进行调整
                        uploaded_file_path = f"/temp/{object_name}"
                
                if uploaded_file_path:  # 找到有效的文件路径就停止
                    break
        
        # 3. 如果有上传的文件，就强制检索该文件的内容
        relevant_docs_text = ""
        if uploaded_file_path and rag_course:
            logger.info(f"检测到用户刚刚上传了文件: {uploaded_file_name}，正在强制检索其内容...")
            
            try:
                # 构建元数据过滤条件
                # 注意：这里需要根据向量库中实际存储的 source 字段格式进行调整
                metadata_filter = {"source": uploaded_file_path}
                
                # 强制检索：设置一个较低的 score_threshold 和较大的 k 值
                retriever = rag_course.as_retriever(
                    search_kwargs={
                        "k": 5,  # 检索前5个最相关的块
                        "score_threshold": 0.3,  # 设置一个较低的阈值
                        "search_type": "similarity",  # 使用相似度搜索
                        "filter": metadata_filter  # 关键：只检索该文件的内容
                    }
                )
                
                # 执行检索
                docs = retriever.invoke(query)
                
                # 将检索到的文档内容拼接成一个字符串
                if docs:
                    relevant_docs_text = "\n\n".join([doc.page_content for doc in docs])
                    logger.info(f"成功检索到 {len(docs)} 个与文件 {uploaded_file_name} 相关的文档块")
                else:
                    logger.warning(f"未在向量库中找到文件 {uploaded_file_name} 的内容")
                    
            except Exception as e:
                logger.error(f"强制检索文件内容失败: {e}", exc_info=True)
        
        # 4. 构建完整的上下文
        # 将聊天历史转换为字符串
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        
        # 合并聊天历史和检索到的文档内容
        full_context = f"{history_text}\n\n{relevant_docs_text}"
        
        return full_context
        
    except Exception as e:
        logger.error(f"获取相关上下文失败: {e}", exc_info=True)
        return ""

# ==================== 当前会话ID管理 (新增) ====================

# 使用线程本地存储来保存当前会话ID
_thread_local = threading.local()

def set_current_thread_id(thread_id: str):
    """
    设置当前线程的会话ID
    
    Args:
        thread_id: 会话ID
    """
    _thread_local.current_thread_id = thread_id
    logger.debug(f"设置当前会话ID: {thread_id}")

def get_current_thread_id() -> str:
    """
    获取当前线程的会话ID
    
    Returns:
        str: 当前会话ID，如果未设置则返回 "default"
    """
    if not hasattr(_thread_local, 'current_thread_id') or _thread_local.current_thread_id is None:
        return "default"
    return _thread_local.current_thread_id

def clear_current_thread_id():
    """清除当前线程的会话ID"""
    if hasattr(_thread_local, 'current_thread_id'):
        _thread_local.current_thread_id = None

# ==================== 文件名映射管理 (新增) ====================

def save_file_mapping(original_name: str, object_name: str, thread_id: str = None) -> bool:
    """
    保存文件名映射关系（原始文件名 -> MinIO 对象名）
    
    Args:
        original_name: 用户上传时的原始文件名（如 "novl.txt"）
        object_name: MinIO 中的对象名（如 "novl_1776344094.txt"）
        thread_id: 所属会话ID（可选）
        
    Returns:
        是否保存成功
    """
    try:
        with get_chat_db_connection() as conn:
            cursor = conn.cursor()
            
            # 检查表是否存在，不存在则创建
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_name TEXT NOT NULL,
                object_name TEXT NOT NULL,
                thread_id TEXT,
                upload_time TEXT NOT NULL,
                UNIQUE(original_name, thread_id)
            )
            """)
            
            # 插入或更新映射
            upload_time = datetime.now().isoformat()
            cursor.execute("""
            INSERT OR REPLACE INTO file_mappings (original_name, object_name, thread_id, upload_time)
            VALUES (?, ?, ?, ?)
            """, (original_name, object_name, thread_id, upload_time))
            
            conn.commit()
            logger.info(f"保存文件映射: {original_name} -> {object_name}")
            return True
            
    except Exception as e:
        logger.error(f"保存文件映射失败: {e}", exc_info=True)
        return False

def get_object_name_by_original(original_name: str, thread_id: str = None) -> Optional[str]:
    """
    根据原始文件名获取 MinIO 对象名
    
    Args:
        original_name: 用户输入的文件名（如"成绩.txt"）
        thread_id: 会话ID（用于区分不同会话下同名文件）
        
    Returns:
        MinIO 对象名（如"成绩_1776341538.txt"），如果未找到则返回None
    """
    try:
        with get_chat_db_connection() as conn:
            cursor = conn.cursor()
            
            if thread_id:
                # 优先在当前会话中查找
                cursor.execute("""
                SELECT object_name FROM file_mappings 
                WHERE original_name = ? AND thread_id = ?
                ORDER BY upload_time DESC LIMIT 1
                """, (original_name, thread_id))
            else:
                # 如果没有指定 thread_id，查找最近的一个
                cursor.execute("""
                SELECT object_name FROM file_mappings 
                WHERE original_name = ?
                ORDER BY upload_time DESC LIMIT 1
                """, (original_name,))
            
            result = cursor.fetchone()
            return result["object_name"] if result else None
            
    except Exception as e:
        logger.error(f"查询文件映射失败: {e}", exc_info=True)
        return None

def list_recent_files(thread_id: str = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    列出最近上传的文件
    
    Args:
        thread_id: 会话ID
        limit: 返回数量
        
    Returns:
        文件列表，包含原始文件名和MinIO对象名
    """
    try:
        with get_chat_db_connection() as conn:
            cursor = conn.cursor()
            
            if thread_id:
                cursor.execute("""
                SELECT original_name, object_name, upload_time FROM file_mappings 
                WHERE thread_id = ?
                ORDER BY upload_time DESC LIMIT ?
                """, (thread_id, limit))
            else:
                cursor.execute("""
                SELECT original_name, object_name, upload_time FROM file_mappings 
                ORDER BY upload_time DESC LIMIT ?
                """, (limit,))
            
            results = cursor.fetchall()
            return [
                {
                    "original_name": row["original_name"],
                    "object_name": row["object_name"],
                    "upload_time": row["upload_time"]
                }
                for row in results
            ]
    except Exception as e:
        logger.error(f"列出最近文件失败: {e}", exc_info=True)
        return []

if __name__ == "__main__":
    # 测试聊天历史数据库
    init_chat_db()
    create_default_session()
    
    # 尝试修复数据库
    repair_database()
    
    # 测试内存检查点
    asyncio.run(test_saver())
