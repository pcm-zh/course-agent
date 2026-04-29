"""
课程智能体核心入口模块（课程问答助手 + 专业领域顾问）
设计理念：
1. 模块化整合：统一管理LLM模型、工具集、会话记忆、向量数据库等核心组件
2. 多模态交互：支持命令行交互式对话，兼容FastAPI后端扩展
3. 会话持久化：基于SQLite实现多线程/多用户会话记忆，支持会话统计和清理
4. 标准化处理：统一响应格式、错误处理、日志记录，提升系统稳定性
5. 可扩展性：预留向量数据库、嵌入模型集成接口

核心功能：
- 初始化并整合LLM模型、课程问答工具、会话记忆系统、向量数据库
- 提供命令行交互式对话界面，支持FastAPI后端集成
- 支持多线程/多用户会话管理，适配多用户访问场景
- 提供会话统计和清理功能，便于系统维护
- 标准化响应处理和友好的终端输出
- 意图识别与Fallback机制
"""

# ==================== 系统基础模块导入 ====================
import os
import sys
import argparse
import json
import re
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
from functools import lru_cache
from enum import Enum


# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ==================== LangChain/LangGraph核心模块 ====================
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# ==================== 项目内部核心组件导入 ====================
from .memory_sqlite import (
    get_sqlite_saver,
    get_checkpoint_count,
    list_threads,
    clear_thread
)
from .config import Config
from .llms import get_chat_model
from .tools import get_tools
from .models import (
    Context, ResponseFormat, Message, Conversation
)
from .logger import LoggerManager

# ==================== 全局初始化配置 ====================
logger = LoggerManager.get_logger(__name__)

# ==================== 核心组件初始化 ====================
try:
    llm_chat = get_chat_model()
    logger.info(f"已加载聊天模型: {Config.DEFAULT_LLM_PROVIDER}")

    tools = get_tools()
    logger.info(f"已注册工具集: {[t.name for t in tools]}")

except Exception as e:
    logger.error(f"核心组件初始化失败: {str(e)}", exc_info=True)
    raise RuntimeError(f"无法初始化模型/工具/向量数据库: {str(e)}") from e

# ==================== 意图分类枚举 ====================
class IntentType(Enum):
    """意图类型枚举"""
    COURSE_QA = "COURSE_QA"  # 课程问答
    FILE_OPERATION = "FILE_OPERATION"  # 文件操作
    CHITCHAT = "CHITCHAT"  # 闲聊
    PROFESSIONAL_CONSULTING = "PROFESSIONAL_CONSULTING"  # 专业咨询
    UNKNOWN = "UNKNOWN"  # 未知/其他

# ==================== 意图识别模块 ====================
class IntentClassifier:
    """意图识别与分类器"""
    
    # 使用类变量存储意图类别，避免重复创建
    INTENT_CATEGORIES = {
        IntentType.COURSE_QA: {
            "desc": "课程问答",
            "definition": "用户询问关于特定课程的内容、知识点、技术概念、编程语言特性、算法、数据结构、考试或作业等信息。包括但不限于课程中涉及的技术主题、编程概念、理论知识和实践内容。",
            "examples": [
                "数据结构课程的主要内容是什么？",
                "高等数学的期末考试范围是什么？",
                "大学物理的作业什么时候截止？",
                "这门课程需要什么先修知识？",
                "如何学习这门课程？",
                "Python的异常处理的内容是什么？",
                "解释一下面向对象编程的概念",
                "什么是递归？",
                "数据库的事务是什么？"
            ],
            "response_style": "结合课程内容，提供专业解答和学习建议",
            "tool_preference": "rag_course",
            "keywords": ["课程", "课件", "知识点", "大纲", "习题", "作业", "考试", "成绩", "教学", "学习", "章节", "内容", "资料", "教材", "讲义", "PPT", "PDF", "概念", "原理", "方法", "技术", "应用", "实现", "编程", "代码", "算法", "数据结构", "数据库", "网络", "操作系统", "异常", "错误", "处理", "函数", "类", "对象", "继承", "多态", "封装", "接口", "抽象", "设计模式"]
        },
        IntentType.FILE_OPERATION: {
            "desc": "文件操作",
            "definition": "用户请求读取、写入、总结或分析特定文件的内容。",
            "examples": [
                "帮我总结一下这个PDF文件的内容",
                "读取这个Word文档并提取关键信息",
                "比较这两个文件的差异"
            ],
            "response_style": "结合文件内容，提供专业分析和学习建议",
            "tool_preference": "根据文件类型选择",
            "keywords": ["文件", "文档", "PDF", "Word", "Excel", "PPT", "读取", "总结", "分析", "比较"]
        },
        IntentType.CHITCHAT: {
            "desc": "闲聊",
            "definition": "日常问候、情感表达或与业务无关的通用对话。",
            "examples": [
                "你好",
                "今天天气怎么样？",
                "你叫什么名字？",
                "你是谁？",
                "谢谢你的帮助"
            ],
            "response_style": "保持专业身份，从课程和专业角度提供见解",
            "tool_preference": "直接回答",
            "keywords": ["你好", "天气", "名字", "谁", "谢谢", "再见", "早上好", "晚上好"]
        },
        IntentType.PROFESSIONAL_CONSULTING: {
            "desc": "专业咨询",
            "definition": "涉及特定领域（如编程、医学、法律）的深度技术问题，且问题内容超出常规课程范围，需要检索最新的外部知识库或专业资料。注意：如果问题涉及课程中已教授的基础概念、技术原理或编程语言特性，应优先归类为COURSE_QA。",
            "examples": [
                "Python 3.12版本有哪些新特性？",
                "最新的医学研究关于COVID-19有什么新发现？",
                "最新的法律法规关于数据隐私有什么变化？",
                "如何设计一个支持千万级用户的分布式数据库架构？",
                "当前AI领域最前沿的研究方向是什么？"
            ],
            "response_style": "结合专业知识和课程内容，提供深度解答",
            "tool_preference": "tavily_search",
            "keywords": ["最新", "研究", "前沿", "趋势", "新闻", "发布", "版本", "更新", "升级", "行业", "市场", "动态", "发展"]
        },
        IntentType.UNKNOWN: {
            "desc": "未知/其他",
            "definition": "无法明确归类的请求，或者模型置信度极低的情况。",
            "examples": [
                "你能做什么？",
                "介绍一下你自己",
                "系统是如何工作的？"
            ],
            "response_style": "保持专业身份，从课程和专业角度提供见解",
            "tool_preference": "尝试使用最相关的工具，或从课程和专业角度回答",
            "keywords": ["什么", "谁", "怎么", "为什么", "如何", "介绍"]
        }
    }

    # 预编译正则表达式，提高性能
    JSON_PATTERN = re.compile(r'\{.*\}', re.DOTALL)
    
    # 预生成类别提示字符串，避免重复生成
    @property
    def CATEGORY_PROMPT_STRING(self) -> str:
        """生成类别提示字符串"""
        return "\n".join([
            f"{k.value}: {v['definition']}\n  示例: {', '.join(v['examples'][:2])}" 
            for k, v in self.INTENT_CATEGORIES.items()
        ])
    
    # 意图分类提示模板
    INTENT_CLASSIFICATION_PROMPT = """
你是一个专业的意图分类助手，服务于课程咨询顾问+专业领域助手系统。请根据用户输入，判断其属于以下哪一类意图。

{categories}

请严格按照 JSON 格式输出，包含以下字段：
1. "intent": 预测的意图类别（必须是上述定义的 Key 之一）。
2. "confidence": 置信度（0.0 到 1.0 之间的浮点数）。
3. "reason": 简短的判断理由。
4. "suggested_response_style": 建议的回答风格（参考意图类别的response_style）。

判断原则：
1. 优先考虑用户问题的核心意图，而非表面关键词
2. 如果问题涉及多个意图，选择最主要的那个
3. 置信度应基于问题与意图定义的匹配程度
4. 如果问题不明确或模糊，降低置信度并选择UNKNOWN

用户输入：{user_input}

JSON 输出：
"""
    
    def __init__(self, llm_client, confidence_threshold=0.7):
        """
        初始化意图分类器
        
        Args:
            llm_client: LLM客户端实例
            confidence_threshold: 置信度阈值，低于此值将触发Fallback机制
        """
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold
        
        # 缓存分类结果，避免重复计算
        self._classification_cache = {}
    
    def _clean_json_response(self, response_text: str) -> str:
        """尝试从模型返回的文本中提取 JSON 字符串"""
        match = self.JSON_PATTERN.search(response_text)
        if match:
            return match.group(0)
        return response_text
    
    def classify(self, user_input: str) -> Dict[str, Any]:
        """
        执行意图分类
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            包含意图类别、置信度、判断理由和建议回答风格的字典
        """
        # 检查缓存
        cache_key = hash(user_input)
        if cache_key in self._classification_cache:
            logger.debug(f"使用缓存的意图分类结果: {user_input}")
            return self._classification_cache[cache_key]
        
        # 检查输入有效性
        if not user_input or not user_input.strip():
            logger.warning("用户输入为空，返回UNKNOWN意图")
            return {
                "intent": IntentType.UNKNOWN,
                "confidence": 0.0,
                "reason": "用户输入为空",
                "suggested_response_style": self.INTENT_CATEGORIES[IntentType.UNKNOWN]["response_style"],
                "tool_preference": self.INTENT_CATEGORIES[IntentType.UNKNOWN]["tool_preference"],
                "response_style": self.INTENT_CATEGORIES[IntentType.UNKNOWN]["response_style"]
            }
        
        prompt = self.INTENT_CLASSIFICATION_PROMPT.format(
            categories=self.CATEGORY_PROMPT_STRING,
            user_input=user_input
        )
        
        try:
            # 调用底层 LLM 进行分类
            from langchain_core.messages import HumanMessage
            response = self.llm_client.invoke([HumanMessage(content=prompt)])
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # 记录原始响应
            logger.debug(f"意图分类原始响应: {response_text}")
            
            # 清理和解析JSON
            cleaned_text = self._clean_json_response(response_text)
            
            try:
                data = json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {e}, 原始文本: {cleaned_text}")
                # 返回UNKNOWN意图
                return {
                    "intent": IntentType.UNKNOWN,
                    "confidence": 0.0,
                    "reason": f"JSON解析失败: {str(e)}",
                    "suggested_response_style": self.INTENT_CATEGORIES[IntentType.UNKNOWN]["response_style"],
                    "tool_preference": self.INTENT_CATEGORIES[IntentType.UNKNOWN]["tool_preference"],
                    "response_style": self.INTENT_CATEGORIES[IntentType.UNKNOWN]["response_style"]
                }
            
            # 验证JSON结构
            required_fields = ["intent", "confidence", "reason", "suggested_response_style"]
            for field in required_fields:
                if field not in data:
                    logger.warning(f"JSON缺少必要字段: {field}")
                    data[field] = "" if field == "reason" else 0.0
            
            # 将字符串形式的意图转换为枚举类型
            intent_str = data.get("intent")
            try:
                intent = IntentType(intent_str)
            except ValueError:
                logger.warning(f"LLM返回了未知意图类别: {intent_str}，重置为 UNKNOWN")
                intent = IntentType.UNKNOWN
                confidence = 0.0
                suggested_response_style = self.INTENT_CATEGORIES[IntentType.UNKNOWN]["response_style"]
            else:
                confidence = float(data.get("confidence", 0))
                # 验证置信度范围
                confidence = max(0.0, min(1.0, confidence))
                reason = data.get("reason", "")
                suggested_response_style = data.get("suggested_response_style", "")
            
            logger.info(f"意图识别 -> 类别: {intent.value}, 置信度: {confidence:.2f}, 理由: {reason}")
            
            # 构建返回结果
            result = {
                "intent": intent,
                "confidence": confidence,
                "reason": reason,
                "suggested_response_style": suggested_response_style,
                "tool_preference": self.INTENT_CATEGORIES[intent]["tool_preference"],
                "response_style": self.INTENT_CATEGORIES[intent]["response_style"]
            }
            
            # 缓存结果
            self._classification_cache[cache_key] = result
            
            return result

        except Exception as e:
            logger.error(f"意图识别异常: {e}", exc_info=True)
            # 返回UNKNOWN意图
            return {
                "intent": IntentType.UNKNOWN,
                "confidence": 0.0,
                "reason": f"识别过程出错: {str(e)}",
                "suggested_response_style": self.INTENT_CATEGORIES[IntentType.UNKNOWN]["response_style"],
                "tool_preference": self.INTENT_CATEGORIES[IntentType.UNKNOWN]["tool_preference"],
                "response_style": self.INTENT_CATEGORIES[IntentType.UNKNOWN]["response_style"]
            }

    def trigger_fallback(self, user_input: str, classification: Dict[str, Any] = None) -> str:
        """
        Fallback机制：当置信度低或意图未知时触发
        
        Args:
            user_input: 用户输入文本
            classification: 意图分类结果，如果提供则使用其中的建议回答风格
            
        Returns:
            Fallback响应文本
        """
        logger.info(f"触发 Fallback 机制: {user_input}")
        
        # 确定回答风格
        response_style = ""
        if classification and "response_style" in classification:
            response_style = classification["response_style"]
        
        # 构建Fallback提示词
        fallback_prompt = f"""
你是一名专业的课程咨询顾问+专业领域助手。请根据以下用户输入，以{response_style}的方式回答。

用户输入：{user_input}

请直接回答用户问题，保持专业身份。
"""
        
        try:
            from langchain_core.messages import HumanMessage
            response = self.llm_client.invoke([HumanMessage(content=fallback_prompt)])
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"抱歉，我暂时无法处理该请求（Fallback失败）：{str(e)}"

# ==================== Agent系统提示词 ====================
SYSTEM_PROMPT = """你是一名专业的**课程问答助手 + 专业领域顾问**。
你的身份定位是课程咨询顾问和专业领域助手的结合体，专门为用户提供课程相关咨询和专业领域支持。

## 你的身份与职责

1. **课程咨询顾问**：
   - 提供课程内容咨询、学习建议、选课指导
   - 解答课程相关问题，包括知识点、作业、考试等
   - 提供学习路径规划和学习策略建议

2. **专业领域助手**：
   - 在特定领域（如编程、医学、法律等）提供专业支持
   - 结合课程内容，提供深度专业解答
   - 帮助用户将理论知识应用于实践

## 回答原则

1. **身份保持**：
   - 始终以"课程咨询顾问+专业领域助手"的身份回答问题
   - 即使面对一般性问题，也要从课程和专业角度提供见解
   - 避免使用通用的AI助手回答方式

2. **回答风格**：
   - 专业、准确、有针对性
   - 结合课程内容和专业知识
   - 提供实用建议和解决方案

3. **自我介绍**：
   - 当被问及"你是谁"时，应该这样回答：
   - "你好！我是课程咨询顾问+专业领域助手，专门为你提供课程相关咨询和专业领域支持。我可以帮助你解答课程问题、提供学习建议、规划学习路径，并在专业领域提供深度支持。有什么我可以帮助你的吗？"

## 可用工具

1. **rag_course**: 用于检索课程知识点、课件内容、课程大纲、习题解析等
2. **tavily_search**: 用于通过网络检索获取最新资讯、行业动态、考试政策、课程公告、天气等
3. **sql_agent**: 用于使用自然语言查询数据库，如学生成绩、选课数据、学习统计、作业完成情况等
4. **files_parser**: 用于解析文件内容，支持PDF、DOCX、PPTX、TXT、HTML、IPYNB等格式

## 意图识别与工具选择

根据用户问题的意图类型，选择合适的工具：

### 1. 课程问答 (COURSE_QA)
当用户询问关于特定课程的内容、安排、考试或作业等信息时：
- **首选工具**: rag_course
- **适用场景**: 课程知识点、课件、大纲、习题、课程安排、考试时间、作业要求等
- **回答重点**: 结合课程内容，提供专业解答和学习建议
- **重要提示**: 
    - 使用rag_course工具时，系统会自动检索相关课程资料
    - 检索到的参考资料会自动格式化并传递给LLM
    - 必须基于检索到的参考资料回答，不要编造信息
    - 如果参考资料不足，可以适当补充但要明确说明
- **示例**:
  - "数据结构课程的主要内容是什么？"
  - "高等数学的期末考试范围是什么？"
  - "大学物理的作业什么时候截止？"

### 2. 文件操作 (FILE_OPERATION)
当用户请求读取、写入、总结或分析特定文件的内容时：
- **首要任务**: 必须调用 `files_parser` 工具提取文件内容。
- **参数处理**: 
    - 如果用户提供的文件名（如 "成绩.txt"）是最近上传的文件，请直接使用该文件名作为 `file_path` 参数。
    - 如果文件名包含时间戳（如 "成绩_1776341538.txt"），请使用完整文件名。
    - 系统会自动将文件名映射到 MinIO 中的对象路径或本地临时路径，工具内部会处理路径转换。
- **回答重点**: 
    - 首先展示工具返回的文件内容摘要（如果工具返回了内容）。
    - 然后结合文件内容，提供专业分析和学习建议。
- **重要提示**: 
    - 当用户请求分析特定文件时，必须使用 `files_parser` 工具提取文件内容。
    - 不要尝试猜测文件内容或凭空回答。
    - 如果工具调用失败或返回错误，请明确告知用户，并建议检查文件名或重新上传。
- **示例**:
  - 用户: "读取成绩.txt的内容"
  - 助手: [调用 files_parser 工具, file_path="成绩.txt"] 
    - 工具返回: "文件内容已成功提取，共 150 个字符。"
  - 助手回答: "根据文件内容，成绩.txt 记录了以下信息..."


### 3. 专业咨询 (PROFESSIONAL_CONSULTING)
当用户涉及特定领域（如编程、医学、法律）的深度技术问题时：
- **首选工具**: tavily_search
- **适用场景**: 获取最新行业动态、专业领域知识、技术解决方案等
- **回答重点**: 结合专业知识和课程内容，提供深度解答
- **示例**:
  - "Python中如何优化内存使用？"
  - "最新的医学研究关于COVID-19有什么新发现？"
  - "最新的法律法规关于数据隐私有什么变化？"

### 4. 闲聊 (CHITCHAT)
当用户进行日常问候、情感表达或与业务无关的通用对话时：
- **处理方式**: 从课程和专业角度回应
- **适用场景**: 问候、情感表达、闲聊等
- **回答重点**: 即使是闲聊，也要保持专业身份，从课程和专业角度提供见解
- **示例**:
  - "你好" → "你好！我是课程咨询顾问+专业领域助手，有什么课程或专业问题我可以帮助你吗？"
  - "今天天气怎么样？" → "作为课程咨询顾问，我建议你关注天气变化，合理安排学习计划。根据最新天气信息..."
  - "你叫什么名字？" → "我是课程咨询顾问+专业领域助手，专门为你提供课程相关咨询和专业领域支持。"

### 5. 未知/其他 (UNKNOWN)
当无法明确归类用户请求，或者模型置信度极低时：
- **处理方式**: 尝试使用最相关的工具，或从课程和专业角度回答
- **适用场景**: 无法明确归类的请求
- **回答重点**: 保持专业身份，从课程和专业角度提供见解
- **示例**:
  - "你能做什么？" → "作为课程咨询顾问+专业领域助手，我可以帮助你解答课程问题、提供学习建议、规划学习路径，并在专业领域提供深度支持。"
  - "介绍一下你自己" → "你好！我是课程咨询顾问+专业领域助手，专门为你提供课程相关咨询和专业领域支持。我可以帮助你解答课程问题、提供学习建议、规划学习路径，并在专业领域提供深度支持。有什么我可以帮助你的吗？"

## 回答指南

1. **身份强调**:
   - 在回答中始终强调自己的身份是"课程咨询顾问+专业领域助手"
   - 即使面对一般性问题，也要从课程和专业角度提供见解
   - 避免使用通用的AI助手回答方式

2. **直接回答**:
   - 请直接回答用户问题，不要添加不必要的开场白或结束语
   - 但在回答中要体现自己的专业身份

3. **工具调用**:
   - 需要工具时自动调用对应工具，不要询问用户是否需要使用工具
   - 在回答中明确指出使用了什么工具获取信息
   - **重要**: 当用户请求分析特定文件时，必须使用files_parser工具提取文件内容

4. **多工具使用**:
   - 对于复杂问题，可以按需调用多个工具，整合多个来源的信息
   - 在回答中明确指出整合了哪些来源的信息

5. **信息来源**:
   - 引用信息来源时，明确指出是来自课程知识库、网络搜索还是数据库查询
   - 结合课程内容，提供专业分析和学习建议

6. **不确定性**:
   - 对于不确定的信息，请明确表示，并建议用户通过其他渠道验证
   - 提供相关的学习资源和参考资料

## 示例对话

**用户**: "你是谁？"
**助手**: "你好！我是课程咨询顾问+专业领域助手，专门为你提供课程相关咨询和专业领域支持。我可以帮助你解答课程问题、提供学习建议、规划学习路径，并在专业领域提供深度支持。有什么我可以帮助你的吗？"

**用户**: "数据结构课程的主要内容是什么？"
**助手**: [使用rag_course工具] 作为课程咨询顾问，我为你查询了数据结构课程的相关内容。数据结构课程主要内容包括：数组、链表、栈、队列、树、图等基本数据结构，以及排序、查找等算法。这些内容是计算机科学的基础，对于理解更高级的算法和数据结构至关重要。建议你按照课程的安排，逐步学习这些内容，并结合实践加深理解。

**用户**: "最新的Python版本有什么新特性？"
**助手**: [使用tavily_search工具] 作为专业领域助手，我为你查询了最新的Python版本信息。Python 3.11版本引入了一些新特性，包括：更快的解释器、更详细的错误信息、新的类型系统特性等。这些改进旨在提高Python的性能和开发体验。对于学习编程课程的同学来说，了解这些新特性有助于跟上技术发展的步伐。

**用户**: "我的数据结构课程成绩是多少？"
**助手**: [使用sql_agent工具] 作为课程咨询顾问，我为你查询了你的课程成绩。根据数据库查询结果，你的数据结构课程成绩是85分。这是你在2023年秋季学期选修的课程，由张教授授课。这个成绩表明你对数据结构的基本概念有了较好的掌握，建议继续深入学习，提高实践能力。

**用户**: "简历模板华测-安安提供_20260412020037_008546.pdf的主要内容"
**助手**: [使用files_parser工具] 作为课程咨询顾问，我为你解析了该PDF文件的内容。这份简历模板主要包含以下几个部分：1. 个人信息：包括姓名、联系方式、教育背景等基本信息；2. 教育经历：详细列出了从高中到大学的学历信息，包括学校、专业、时间等；3. 实习经历：描述了在不同公司的实习经历，包括职位、工作内容和成果；4. 技能特长：列出了掌握的技能和特长，包括语言能力、计算机技能等；5. 荣誉奖项：展示了获得的各种荣誉和奖项。这份简历模板结构清晰，内容全面，适合作为求职参考。如果你需要修改或完善简历，我可以提供一些建议。

请根据用户的问题，选择最合适的工具，以"课程咨询顾问+专业领域助手"的身份，提供准确、有用的回答。
"""

# 全局Agent与记忆
_agent_instance = None
_checkpointer = None
_intent_classifier = None  # 全局意图分类器实例

# ==================== Agent实例管理 ====================
@lru_cache(maxsize=1)
def get_agent(force_rebuild: bool = False):
    """
    获取Agent实例（单例模式）
    整合LLM、工具、系统提示、会话记忆
    
    Args:
        force_rebuild: 是否强制重建Agent实例
        
    Returns:
        Agent实例
    """
    global _agent_instance, _checkpointer, _intent_classifier

    if _agent_instance is not None and not force_rebuild:
        return _agent_instance

    try:
        _checkpointer = get_sqlite_saver()
        
        # 初始化意图分类器
        if _intent_classifier is None:
            _intent_classifier = IntentClassifier(llm_client=llm_chat)
            logger.info("意图分类器初始化完成")

        # 尝试多种导入方式，兼容不同版本的 LangChain
        logger.info("开始初始化课程问答Agent...")
        
        try:
            # 尝试从 langchain.agents 导入（旧版本）
            from langchain.agents import AgentExecutor
            logger.info("使用 langchain.agents.AgentExecutor")
        except ImportError:
            try:
                # 尝试从 langchain_core.agents 导入（新版本）
                from langchain_core.agents import AgentExecutor
                logger.info("使用 langchain_core.agents.AgentExecutor")
            except ImportError:
                try:
                    # 尝试从 langchain.agents.agent 导入
                    from langchain.agents.agent import AgentExecutor
                    logger.info("使用 langchain.agents.agent.AgentExecutor")
                except ImportError:
                    # 如果都失败了，使用 LangGraph 的方式
                    logger.warning("无法导入 AgentExecutor，将使用 LangGraph 方式")
                    # 使用 LangGraph 的 Agent
                    _agent_instance = create_agent(
                        model=llm_chat,
                        system_prompt=SYSTEM_PROMPT,
                        tools=tools,
                        checkpointer=_checkpointer,
                    )
                    logger.info("课程问答Agent初始化完成（LangGraph方式）")
                    return _agent_instance
        
        # 如果成功导入 AgentExecutor，则使用它来包装 agent
        # 创建基础 agent
        base_agent = create_agent(
            model=llm_chat,
            system_prompt=SYSTEM_PROMPT,
            tools=tools,
            checkpointer=_checkpointer,
        )
        
        # 创建 AgentExecutor 来执行工具调用
        _agent_instance = AgentExecutor(
            agent=base_agent,
            tools=tools,
            verbose=True,  # 启用详细日志
            handle_parsing_errors=True,  # 处理解析错误
            max_iterations=10,  # 最大迭代次数
            
            handle_tool_errors=True,  # 处理工具错误
            early_stopping_method="generate",  # 早期停止方法
        )

        logger.info("课程问答Agent初始化完成")
        return _agent_instance

    except Exception as e:
        logger.error(f"Agent初始化失败: {str(e)}", exc_info=True)
        raise RuntimeError(f"无法初始化课程智能体: {str(e)}") from e

# ==================== 获取意图分类器实例 ====================
def get_intent_classifier():
    """
    获取意图分类器实例（单例模式）
    确保在调用前 Agent 已经初始化（因为 Classifier 依赖 Agent 的 LLM）
    
    Returns:
        意图分类器实例
        
    Raises:
        RuntimeError: 如果意图分类器初始化失败
    """
    global _intent_classifier
    
    if _intent_classifier is None:
        # 如果未初始化，尝试通过 get_agent 触发初始化
        try:
            get_agent()
        except Exception as e:
            logger.error(f"无法初始化意图分类器 (Agent初始化失败): {e}")
            raise RuntimeError("意图分类器初始化失败，请检查 LLM 配置") from e
            
    return _intent_classifier

# ==================== 响应处理 ====================

def process_response(agent_response) -> ResponseFormat:
    """
    处理Agent响应，提取答案和工具调用信息
    增强版：能够正确处理包含工具调用的响应
    """
    # 初始化响应格式
    response = ResponseFormat(answer="")
    
    # 处理不同类型的响应
    if isinstance(agent_response, dict):
        # 提取答案
        if "output" in agent_response:
            response.answer = agent_response["output"]
        elif "messages" in agent_response and len(agent_response["messages"]) > 0:
            # 从最后一条消息中提取答案
            last_message = agent_response["messages"][-1]
            if isinstance(last_message, dict) and "content" in last_message:
                response.answer = last_message["content"]
            elif hasattr(last_message, "content"):
                response.answer = last_message.content
        else:
            # 如果没有标准字段，尝试直接使用整个响应
            response.answer = str(agent_response)
        

        # 设置置信度
        if "confidence" in agent_response:
            response.confidence = agent_response["confidence"]
        else:
            response.confidence = 0.5  # 默认置信度
    
    elif isinstance(agent_response, str):
        # 如果响应是字符串，直接作为答案
        response.answer = agent_response
        response.confidence = 0.5  # 默认置信度
    
    elif hasattr(agent_response, 'content'):
        # 如果响应有content属性，直接使用
        response.answer = agent_response.content
        response.confidence = 0.5  # 默认置信度
    
    else:
        # 其他情况，转换为字符串
        response.answer = str(agent_response)
        response.confidence = 0.5  # 默认置信度
    
    # 检查答案是否为空
    if not response.answer or not response.answer.strip():
        logger.warning("处理后的答案为空")
        response.answer = "抱歉，我无法回答这个问题。"
        response.error = "答案为空"
    
    return response

def display_response(response: ResponseFormat, verbose: bool = False):
    """
    终端友好显示回答
    
    Args:
        response: 标准化的响应格式
        verbose: 是否显示详细信息
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

    print("-" * 60)

# ==================== 对话管理 ====================
def run_conversation(
    thread_id: str = None,
    user_id: str = None,
    verbose: bool = False,
    max_turns: int = None,
    auto_exit: bool = False
):
    """
    运行交互式课程问答对话
    
    Args:
        thread_id: 会话ID，默认为Config.DEFAULT_THREAD_ID
        user_id: 用户ID，默认为Config.DEFAULT_USER_ID
        verbose: 是否显示详细信息
        max_turns: 最大对话轮次，默认为Config.MAX_TURNS
        auto_exit: 达到最大轮次后是否自动退出
    """
    # 如果未传入 ID，使用 Config 中的默认值
    if thread_id is None:
        thread_id = Config.DEFAULT_THREAD_ID
    if user_id is None:
        user_id = Config.DEFAULT_USER_ID
    if max_turns is None:
        max_turns = Config.MAX_TURNS

    try:
        agent = get_agent()
        config = {"configurable": {"thread_id": thread_id}}

        # 使用 Config 中的颜色代码
        print(f"\n{Config.ANSI_SYSTEM}===== 课程问答助手（用户ID：{user_id}）====={Config.ANSI_END}")
        print(f"{Config.ANSI_SYSTEM}输入 '退出' 或 'exit' 结束对话{Config.ANSI_END}\n")

        turn_count = 0
        while True:
            if max_turns is not None and turn_count >= max_turns:
                if auto_exit:
                    logger.info(f"达到最大轮次 {max_turns}，自动退出")
                    break
                else:
                    print(f"\n{Config.ANSI_WARNING}已达最大轮次 {max_turns}，输入 '继续' 继续对话{Config.ANSI_END}")
                    # 使用 Config 中的颜色代码
                    user_input = input(f"{Config.ANSI_USER}用户:{Config.ANSI_END} ").strip()
                    if user_input.lower() not in ["继续", "continue"]:
                        logger.info("对话结束")
                        break
                    turn_count = 0
                    continue

            # 使用 Config 中的颜色代码
            user_input = input(f"{Config.ANSI_USER}用户:{Config.ANSI_END} ").strip()

            if user_input.lower() in ["退出", "exit"]:
                logger.info("对话结束")
                break
            if not user_input:
                continue

            try:
                logger.info(f"用户提问: {user_input}")

                # --- 意图识别逻辑 ---
                global _intent_classifier
                if _intent_classifier:
                    classification = _intent_classifier.classify(user_input)
                    
                    # 如果置信度过低或意图未知，触发 Fallback
                    
                    if classification['confidence'] < _intent_classifier.confidence_threshold or \
                        classification['intent'] == IntentType.UNKNOWN:
                        logger.info(f"意图置信度不足 ({classification['confidence']}) 或未知，触发 Fallback")
                        logger.info(f"用户输入: {user_input}")
                        logger.info(f"分类结果: {classification}")
                        fallback_answer = _intent_classifier.trigger_fallback(user_input, classification)
                        logger.info(f"Fallback回答: {fallback_answer}")
                        
                        # 不直接返回，而是继续执行Agent流程
                        # 这样可以确保即使意图不明确，也能利用Agent的工具能力
                        response = agent.invoke(
                            input={"messages": [{"role": "user", "content": user_input}]},
                            config=config
                        )
                        structured = process_response(response)
                        display_response(structured, verbose)
                        turn_count += 1
                        continue

                    # 如果意图是闲聊，也可以选择直接回答，不进 Agent
                    if classification['intent'] == IntentType.CHITCHAT:
                        logger.info("识别为闲聊意图，触发 Fallback (直接LLM回答)")
                        chitchat_answer = _intent_classifier.trigger_fallback(user_input, classification)
                        print(f"\n{Config.ANSI_AGENT}课程助手回复:{Config.ANSI_END} {chitchat_answer}")
                        print("-" * 60)
                        turn_count += 1
                        continue
                    
                    # 如果意图是文件操作，确保调用文件解析工具
                    if classification['intent'] == IntentType.FILE_OPERATION:
                        logger.info("识别为文件操作意图，将调用文件解析工具")
                        # 这里不触发 Fallback，而是让 Agent 处理
                        # Agent 会根据系统提示词选择合适的工具（包括文件解析工具）
                        # 确保系统提示词中明确指示了文件操作应该使用文件解析工具

                # --- 正常 Agent 流程 ---
                response = agent.invoke(
                    input={"messages": [{"role": "user", "content": user_input}]},
                    config=config
                )

                structured = process_response(response)
                display_response(structured, verbose)

                turn_count += 1

            except Exception as e:
                error_msg = f"回答出错：{str(e)}"
                logger.error(error_msg, exc_info=True)
                # 使用 Config 中的颜色代码
                print(f"\n{Config.ANSI_ERROR}错误:{Config.ANSI_END} {error_msg}")
                print("-" * 60)

    except Exception as e:
        error_msg = f"对话系统异常: {str(e)}"
        logger.error(error_msg, exc_info=True)
        # 使用 Config 中的颜色代码
        print(f"\n{Config.ANSI_ERROR}系统错误:{Config.ANSI_END} {error_msg}")
        print("-" * 60)

# ==================== 统计功能 ====================
def show_stats():
    """显示会话统计信息"""
    try:
        threads = list_threads()
        # 使用 Config 中的颜色代码
        print(f"\n{Config.ANSI_INFO}===== 课程问答会话统计 ====={Config.ANSI_END}")
        print(f"{Config.ANSI_INFO}总会话数:{Config.ANSI_END} {len(threads)}")

        for thread_id in threads[:10]:
            count = get_checkpoint_count(thread_id)
            print(f"{Config.ANSI_INFO}会话 {thread_id}:{Config.ANSI_END} {count} 条记录")

        if len(threads) > 10:
            print(f"{Config.ANSI_INFO}... 还有 {len(threads) - 10} 个会话{Config.ANSI_END}")

        print("-" * 60)

    except Exception as e:
        error_msg = f"获取统计失败: {str(e)}"
        logger.error(error_msg, exc_info=True)
        # 使用 Config 中的颜色代码
        print(f"\n{Config.ANSI_ERROR}错误:{Config.ANSI_END} {error_msg}")

# ==================== 主函数 ====================
def main():
    """主函数，处理命令行参数并启动相应的功能"""
    parser = argparse.ArgumentParser(description="课程问答智能助手")
    # 默认值从 Config 读取
    parser.add_argument("--thread-id", type=str, default=Config.DEFAULT_THREAD_ID, help="会话ID")
    parser.add_argument("--user-id", type=str, default=Config.DEFAULT_USER_ID, help="用户ID")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    parser.add_argument("--max-turns", type=int, default=Config.MAX_TURNS, help="最大对话轮次")
    parser.add_argument("--auto-exit", action="store_true", help="达到轮次自动退出")
    parser.add_argument("--stats", action="store_true", help="显示统计")
    parser.add_argument("--clear-thread", type=str, default=None, help="清除会话")

    args = parser.parse_args()

    try:
        if args.stats:
            show_stats()
            return

        if args.clear_thread:
            clear_thread(args.clear_thread)
            # 使用 Config 中的颜色代码
            print(f"{Config.ANSI_OKGREEN}已清除会话 {args.clear_thread}{Config.ANSI_END}")
            return

        run_conversation(
            thread_id=args.thread_id,
            user_id=args.user_id,
            verbose=args.verbose,
            max_turns=args.max_turns,
            auto_exit=args.auto_exit
        )

    except KeyboardInterrupt:
        logger.info("用户中断对话")
        # 使用 Config 中的颜色代码
        print(f"\n{Config.ANSI_WARNING}对话已中断{Config.ANSI_END}")
    except Exception as e:
        logger.error(f"运行出错: {str(e)}", exc_info=True)
        # 使用 Config 中的颜色代码
        print(f"\n{Config.ANSI_ERROR}错误:{Config.ANSI_END} {str(e)}")

if __name__ == "__main__":
    main()
