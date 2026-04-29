"""
数据模型定义模块
定义Agent运行上下文和结构化响应的数据类型
核心设计：
1. 使用dataclass装饰器创建结构化数据模型，兼顾易用性和类型安全
2. 实现统一的序列化/反序列化接口（to_dict/from_dict/to_json/from_json）
3. 提供丰富的辅助方法，简化数据操作
4. 内置数据校验和默认值处理逻辑
"""
# 导入数据类相关装饰器和函数
from dataclasses import dataclass, field, asdict
# 导入类型注解，用于字段类型提示
from typing import Optional, Dict, Any, List
# 导入日期时间模块，用于时间戳处理
from datetime import datetime
# 导入JSON模块，用于JSON序列化/反序列化
import json


@dataclass
class Context:
    """
    Agent运行上下文模型
    存储Agent会话的核心上下文信息，用于追踪和管理不同用户/会话的状态
    核心字段：用户ID、线程ID、会话ID、时间戳、元数据
    """
    # 必选字段：用户唯一标识（区分不同用户）
    user_id: str
    # 可选字段：线程ID（LangGraph检查点使用）
    thread_id: Optional[str] = None
    # 可选字段：会话ID（区分同一用户的不同会话）
    session_id: Optional[str] = None
    # 时间戳：默认使用当前时间的ISO格式字符串（如"2026-03-21T10:00:00.123456"）
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # 元数据：存储额外的自定义信息，默认空字典
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        将上下文对象转换为字典（序列化）
        使用dataclasses.asdict确保所有字段正确转换
        
        Returns:
            Dict[str, Any]: 包含所有字段的字典
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Context':
        """
        从字典创建上下文对象（反序列化）
        智能处理字段映射：已知字段直接赋值，未知字段存入metadata
        
        Args:
            data: 包含上下文信息的字典
            
        Returns:
            Context: 实例化的上下文对象
        """
        # 提取字典中与Context字段匹配的键值对
        known_fields = {
            f: data.get(f) 
            for f in cls.__dataclass_fields__.keys() 
            if f in data
        }
        # 将字典中不匹配的字段存入metadata
        metadata = {
            k: v 
            for k, v in data.items() 
            if k not in cls.__dataclass_fields__.keys()
        }
        # 如果有额外字段，更新metadata
        if metadata:
            known_fields['metadata'] = metadata
        # 创建并返回Context实例
        return cls(**known_fields)

    def update_metadata(self, key: str, value: Any) -> None:
        """
        更新元数据字段（便捷方法）
        避免直接操作self.metadata的繁琐
        
        Args:
            key: 元数据键名
            value: 元数据值
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        获取元数据值（便捷方法）
        支持默认值，避免KeyError
        
        Args:
            key: 元数据键名
            default: 默认值（可选）
            
        Returns:
            Any: 元数据值或默认值
        """
        return self.metadata.get(key, default)


@dataclass
class ToolResult:
    """
    工具执行结果模型
    标准化工具调用的返回结果，统一错误处理和结果格式
    """
    # 工具名称（如"tavily_search"、"sql_query"）
    tool_name: str
    # 执行是否成功（布尔值）
    success: bool
    # 执行结果（字符串格式，便于序列化）
    result: str
    # 执行耗时（秒），默认0.0
    execution_time: float = 0.0
    # 错误信息（执行失败时填充），默认None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """将工具结果对象转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolResult':
        """从字典创建工具结果对象"""
        return cls(**data)


@dataclass
class ResponseFormat:
    """
    Agent结构化响应模型
    定义Agent输出的标准化格式，确保返回结果的一致性和可解析性
    包含核心回答、工具使用信息、参考资料、置信度等字段
    """
    # 核心回答内容（必选）
    answer: str
    # 使用的工具名称（可选）
    tool_used: Optional[str] = None
    # 法律参考/依据（可选，适用于医疗/法律等领域）
    legal_references: Optional[str] = None
    # 搜索结果（可选，存储工具返回的搜索信息）
    search_results: Optional[str] = None
    # SQL查询结果（可选，存储数据库查询结果）
    sql_results: Optional[str] = None
    # RAG检索上下文（可选，存储向量检索的相关文档内容）
    rag_context: Optional[str] = None
    # 回答置信度（0.0-1.0），可选
    confidence: Optional[float] = None
    # 错误信息（执行失败时填充），默认None
    error: Optional[str] = None
    # 工具执行结果列表，默认空列表
    tool_results: List[ToolResult] = field(default_factory=list)
    # 元数据，存储额外信息，默认空字典
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        后置初始化方法（dataclass特性）
        自动校验并修正置信度范围，确保在0.0-1.0之间
        """
        if self.confidence is not None:
            # 限制置信度最小值0.0，最大值1.0
            self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> Dict[str, Any]:
        """
        将响应对象转换为字典（深度序列化）
        自动将ToolResult列表转换为字典列表
        
        Returns:
            Dict[str, Any]: 包含所有字段的字典
        """
        data = asdict(self)
        # 递归序列化tool_results中的每个ToolResult对象
        data['tool_results'] = [
            tr.to_dict() if isinstance(tr, ToolResult) else tr 
            for tr in self.tool_results
        ]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResponseFormat':
        """
        从字典创建响应对象（深度反序列化）
        自动将字典列表转换为ToolResult对象列表
        
        Args:
            data: 包含响应信息的字典
            
        Returns:
            ResponseFormat: 实例化的响应对象
        """
        # 处理tool_results字段的反序列化
        if 'tool_results' in data and isinstance(data['tool_results'], list):
            data['tool_results'] = [
                ToolResult.from_dict(tr) if isinstance(tr, dict) else tr 
                for tr in data['tool_results']
            ]
        return cls(**data)

    def to_json(self, indent: int = None) -> str:
        """
        将响应对象转换为JSON字符串
        支持缩进格式化，便于阅读
        
        Args:
            indent: 缩进空格数（可选）
            
        Returns:
            str: JSON格式字符串
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> 'ResponseFormat':
        """
        从JSON字符串创建响应对象
        
        Args:
            json_str: JSON格式字符串
            
        Returns:
            ResponseFormat: 实例化的响应对象
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def add_tool_result(self, tool_result: ToolResult) -> None:
        """
        添加工具执行结果到响应（便捷方法）
        
        Args:
            tool_result: ToolResult对象
        """
        self.tool_results.append(tool_result)

    def get_tool_result(self, tool_name: str) -> Optional[ToolResult]:
        """
        获取指定工具的执行结果
        
        Args:
            tool_name: 工具名称
            
        Returns:
            ToolResult: 匹配的工具结果，无匹配则返回None
        """
        for tr in self.tool_results:
            if tr.tool_name == tool_name:
                return tr
        return None

    def update_metadata(self, key: str, value: Any) -> None:
        """更新响应元数据"""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取响应元数据（支持默认值）"""
        return self.metadata.get(key, default)


@dataclass
class Message:
    """
    消息模型
    定义对话中的单条消息格式，兼容LangChain/LangGraph的消息格式
    """
    # 消息角色（user/assistant/system/tool）
    role: str
    # 消息内容
    content: str
    # 消息时间戳，默认当前时间ISO格式
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # 消息元数据，默认空字典
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """将消息转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息"""
        return cls(**data)

    def to_json(self, indent: int = None) -> str:
        """将消息转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """从JSON字符串创建消息"""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class Conversation:
    """
    对话模型
    整合上下文、消息列表和响应结果，形成完整的对话会话模型
    """
    # 对话上下文
    context: Context
    # 消息列表，默认空列表
    messages: List[Message] = field(default_factory=list)
    # Agent响应结果，可选
    response: Optional[ResponseFormat] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        将对话对象转换为字典（深度序列化）
        递归序列化所有嵌套对象
        
        Returns:
            Dict[str, Any]: 包含所有对话信息的字典
        """
        data = {
            # 序列化上下文
            'context': self.context.to_dict(),
            # 序列化消息列表
            'messages': [msg.to_dict() for msg in self.messages],
            # 序列化响应（处理None情况）
            'response': self.response.to_dict() if self.response else None
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """
        从字典创建对话对象（深度反序列化）
        
        Args:
            data: 包含对话信息的字典
            
        Returns:
            Conversation: 实例化的对话对象
        """
        # 反序列化上下文
        context = Context.from_dict(data.get('context', {}))
        # 反序列化消息列表
        messages = [Message.from_dict(msg) for msg in data.get('messages', [])]
        # 反序列化响应（处理None情况）
        response = ResponseFormat.from_dict(data['response']) if data.get('response') else None
        # 创建并返回Conversation实例
        return cls(context=context, messages=messages, response=response)

    def to_json(self, indent: int = None) -> str:
        """将对话转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> 'Conversation':
        """从JSON字符串创建对话对象"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """
        快捷添加消息到对话
        
        Args:
            role: 消息角色
            content: 消息内容
            metadata: 消息元数据（可选）
        """
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)

    def get_user_messages(self) -> List[Message]:
        """
        筛选获取所有用户消息
        
        Returns:
            List[Message]: 用户消息列表
        """
        return [msg for msg in self.messages if msg.role == "user"]

    def get_assistant_messages(self) -> List[Message]:
        """
        筛选获取所有助手消息
        
        Returns:
            List[Message]: 助手消息列表
        """
        return [msg for msg in self.messages if msg.role == "assistant"]