"""
LLM交互模块
提供与大语言模型交互的统一接口，支持通义千问模型
"""
import os
import json
from typing import Dict, List, Optional, Union, Any, Sequence, Callable, Type
import logging

# ==================== LangChain 基础导入 ====================
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.language_models.llms import BaseLLM as LangChainBaseLLM
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages.tool import ToolCall

# ==================== 配置导入与降级处理 ====================
try:
    from .config import Config, MODEL_CONFIGS
except ImportError:
    class Config:
        DEFAULT_LLM_PROVIDER = "qwen"
        DEFAULT_LLM_MODEL = "qwen-turbo-latest"
        QWEN_API_KEY = ""
        QWEN_CHAT_MODEL = "qwen-turbo-latest"
        QWEN_EMBEDDING_MODEL = "text-embedding-v1"
    
    MODEL_CONFIGS = {
        "qwen": {
            "api_key": Config.QWEN_API_KEY,
            "chat_model": Config.QWEN_CHAT_MODEL,
            "embedding_model": Config.QWEN_EMBEDDING_MODEL,
            "base_url": "https://dashscope.aliyuncs.com/compatible-model/v1"
        }
    }

# ==================== 异常类定义 ====================
class LLMError(Exception):
    """LLM相关异常的基类"""
    pass

class LLMConfigError(LLMError):
    """LLM配置错误异常"""
    pass

class LLMResponseError(LLMError):
    """LLM响应错误异常"""
    pass

# ==================== 日志初始化 ====================
logger = logging.getLogger(__name__)

# ==================== 通义千问LLM实现类 ====================
class QwenLLM(BaseChatModel):
    """
    通义千问模型具体实现类
    继承LangChain的BaseChatModel，实现千问模型的客户端初始化和接口调用逻辑
    """
    
    # 显式声明 Pydantic 字段
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    
    client: Any = None
    default_params: Dict[str, Any] = {}
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        """
        初始化LLM实例
        """
        # 1. 预处理 API Key
        resolved_api_key = api_key
        if not resolved_api_key:
            env_key = f"{model_name.upper().replace('-', '_')}_API_KEY"
            resolved_api_key = os.getenv(env_key, "")
            
            if not resolved_api_key:
                provider = self._get_provider_from_model_name(model_name)
                if provider and provider in MODEL_CONFIGS:
                    resolved_api_key = MODEL_CONFIGS[provider].get("api_key", "")
        
        # 验证配置
        if not resolved_api_key:
            raise LLMConfigError(f"未提供{model_name}的API密钥")
            
        # 2. 准备传递给父类的数据
        init_data = {
            "model_name": model_name,
            "api_key": resolved_api_key,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "top_p": kwargs.get("top_p", 1.0),
        }
        
        # 3. 调用父类初始化
        super().__init__(**init_data)
        
        # 4. 初始化客户端
        self._initialize_client(**kwargs)
        logger.info(f"初始化{model_name}模型成功 (ChatModel模式)")
    
    def _get_provider_from_model_name(self, model_name: str) -> Optional[str]:
        """从模型名称推断提供商"""
        if model_name.startswith("qwen") or model_name.startswith("text-embedding"):
            return "qwen"
        return None
    
    def _initialize_client(self, **kwargs):
        """初始化通义千问客户端"""
        try:
            import dashscope
            dashscope.api_key = self.api_key
            self.client = dashscope
        except ImportError:
            raise LLMConfigError("未安装dashscope库，请使用pip install dashscope安装")
        
        self.default_params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型标识"""
        return "qwen"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """获取模型标识参数"""
        return {
            "model_name": self.model_name,
            "temperature": self.default_params.get("temperature", 0.7),
            "max_tokens": self.default_params.get("max_tokens", 1000),
        }

    
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseTool], BaseTool, Callable]],
        **kwargs: Any,
    ) -> "QwenLLM":
        """
        绑定工具到模型实例。
        """
        formatted_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                formatted_tools.append(tool)
            else:
                formatted_tools.append(convert_to_openai_tool(tool))
        
        self._bound_tools = formatted_tools
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        实现LangChain BaseChatModel要求的抽象方法 _generate。
        """
        generations = []
        qwen_messages = self._convert_messages(messages)
        
        try:
            params = {**self.default_params, **kwargs}
            
            # 检查是否有绑定的工具
            tools_to_call = getattr(self, '_bound_tools', None)
            if tools_to_call:
                params['tools'] = tools_to_call
            
            response = self.client.Generation.call(
                model=self.model_name,
                messages=qwen_messages,
                result_format='message', 
                **params
            )
            
            if response.status_code != 200:
                raise LLMResponseError(f"通义千问API请求失败: {response.message}")
            
            message_content = ""
            tool_calls = []
            
            if hasattr(response, 'output') and hasattr(response.output, 'choices'):
                choice = response.output.choices[0]
                
                # --- 1. 解析 Content ---
                try:
                    if hasattr(choice, 'message'):
                        msg = choice.message
                        # 尝试获取 content
                        if hasattr(msg, 'content'):
                            message_content = msg.content
                        else:
                            # 尝试字典访问
                            try:
                                message_content = msg['content']
                            except (KeyError, TypeError):
                                message_content = ""
                    elif hasattr(choice, 'content'):
                        message_content = choice.content
                except Exception as e:
                    logger.warning(f"解析 content 失败: {e}")
                    message_content = ""

                # --- 2. 解析 Tool Calls
                # 策略：完全避免使用 hasattr 检查 DashScope 对象的动态属性
                # 直接尝试获取数据，捕获 KeyError
                
                raw_tool_calls = None
                
                # 尝试路径 A: choice.message.tool_calls
                if hasattr(choice, 'message'):
                    msg = choice.message
                    try:
                        # 这里直接访问，如果不存在会抛 KeyError，我们捕获它
                        raw_tool_calls = msg.tool_calls
                    except (KeyError, AttributeError):
                        # 如果上面失败，尝试用字典方式获取
                        try:
                            raw_tool_calls = msg['tool_calls']
                        except (KeyError, TypeError):
                            pass
                
                # 尝试路径 B: choice.tool_calls (兼容性兜底)
                if not raw_tool_calls:
                    try:
                        raw_tool_calls = choice.tool_calls
                    except (KeyError, AttributeError):
                        pass

                # --- 3. 转换工具调用格式 ---
                if raw_tool_calls:
                    logger.debug(f"原始 tool_calls 数据: {raw_tool_calls}")
                    for tc in raw_tool_calls:
                        function = None
                        name = None
                        arguments = None
                        
                        # 处理字典格式
                        if isinstance(tc, dict):
                            function = tc.get('function', {})
                        # 处理对象格式
                        elif hasattr(tc, 'function'):
                            function = tc.function
                        
                        # 提取名称和参数
                        if function:
                            if isinstance(function, dict):
                                name = function.get('name')
                                arguments = function.get('arguments')
                            elif hasattr(function, 'name'):
                                name = function.name
                                arguments = function.arguments if hasattr(function, 'arguments') else None
                        
                        # 构造 LangChain ToolCall
                        if name:
                            try:
                                # arguments 通常是 JSON 字符串
                                args_dict = json.loads(arguments) if isinstance(arguments, str) else arguments
                            except:
                                args_dict = {}
                            
                            tool_calls.append(
                                ToolCall(name=name, args=args_dict, id=str(hash(name)))
                            )
                    
                    if tool_calls:
                        logger.info(f"成功解析 {len(tool_calls)} 个工具调用")

            # 构造 AIMessage
            ai_message = AIMessage(content=message_content, tool_calls=tool_calls)
            generations.append(ChatGeneration(message=ai_message))
            
        except Exception as e:
            logger.error(f"生成文本时出错: {str(e)}", exc_info=True)
            raise LLMResponseError(f"生成文本时出错: {str(e)}")
        
        return ChatResult(generations=generations)

    
    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """
        消息格式转换
        """
        converted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = getattr(msg, 'type', "user")
                if role == "human": role = "user"
                elif role == "ai": role = "assistant"
                
            content = msg.content
            converted.append({"role": role, "content": content})
        
        return converted
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """文本生成接口（单次提示词生成）"""
        messages = [HumanMessage(content=prompt)]
        result = self._generate(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
        return result.generations[0].text
    
    def chat(
        self,
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """对话生成接口（多轮对话）"""
        lc_messages = []
        for msg in messages:
            if hasattr(msg, 'type') and isinstance(msg, BaseMessage):
                lc_messages.append(msg)
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=str(msg)))
                
        result = self._generate(lc_messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
        
        if result.generations and len(result.generations) > 0:
            return result.generations[0].text
        
        return ""

    def invoke(self, input: Union[str, List[Dict[str, str]]], config: Optional[Dict[str, Any]] = None) -> str:
        """通用调用接口（兼容LangChain标准）"""
        if isinstance(input, str):
            return self.generate(input)
        elif isinstance(input, list):
            return self.chat(input)
        else:
            raise ValueError(f"不支持的输入类型: {type(input)}")


# ==================== 通义千问嵌入模型实现类 ====================
class QwenEmbeddings(Embeddings):
    """通义千问嵌入模型实现类"""
    MAX_BATCH_SIZE = 25

    def __init__(self, model_name: str = "text-embedding-v1", api_key: Optional[str] = None, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        
        if not self.api_key:
            self.api_key = os.getenv("QWEN_API_KEY", "")
            if not self.api_key:
                provider = "qwen"
                if provider in MODEL_CONFIGS:
                    self.api_key = MODEL_CONFIGS[provider].get("api_key", "")
        
        if not self.api_key:
            raise LLMConfigError("未提供 DashScope API Key")
        
        try:
            import dashscope
            dashscope.api_key = self.api_key
            self.dashscope_module = dashscope
        except ImportError:
            raise LLMConfigError("未安装dashscope库")
            
        logger.info(f"初始化 Qwen 嵌入模型成功: {self.model_name}")

    def _parse_embedding_response(self, resp) -> List[List[float]]:
        """解析DashScope API的响应"""
        status_code = getattr(resp, 'status_code', None)
        if status_code is None and isinstance(resp, dict):
            status_code = resp.get('code')
            
        if status_code != 200:
            message = getattr(resp, 'message', None)
            if message is None and isinstance(resp, dict):
                message = resp.get('message', "未知错误")
            raise LLMResponseError(f"DashScope Embedding API 请求失败 (Code: {status_code}): {message}")

        embeddings = []
        
        if hasattr(resp, 'output'):
            output = resp.output
            if hasattr(output, 'embeddings'):
                for item in output.embeddings:
                    if hasattr(item, 'embedding'):
                        embeddings.append(item.embedding)
                    elif isinstance(item, dict) and 'embedding' in item:
                        embeddings.append(item['embedding'])
            elif isinstance(output, dict) and 'embeddings' in output:
                for item in output['embeddings']:
                    if isinstance(item, dict) and 'embedding' in item:
                        embeddings.append(item['embedding'])
        
        elif isinstance(resp, dict):
            if 'output' in resp and 'embeddings' in resp['output']:
                for item in resp['output']['embeddings']:
                    if isinstance(item, dict) and 'embedding' in item:
                        embeddings.append(item['embedding'])
        
        if not embeddings:
            logger.error(f"无法从响应中解析嵌入向量。响应类型: {type(resp)}")
            raise LLMResponseError("无法从响应中解析嵌入向量")
            
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """计算文档列表的嵌入向量"""
        all_embeddings = []
        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[i : i + self.MAX_BATCH_SIZE]
            try:
                resp = self.dashscope_module.TextEmbedding.call(
                    model=self.model_name,
                    input=batch,
                    text_type="document"
                )
                batch_embeddings = self._parse_embedding_response(resp)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"计算文档嵌入失败: {str(e)}", exc_info=True)
                if isinstance(e, LLMResponseError):
                    raise
                raise LLMResponseError(f"计算文档嵌入失败: {str(e)}")
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """计算单个查询文本的嵌入向量"""
        try:
            resp = self.dashscope_module.TextEmbedding.call(
                model=self.model_name,
                input=text,
                text_type="query"
            )
            embeddings = self._parse_embedding_response(resp)
            if len(embeddings) > 0:
                return embeddings[0]
            else:
                raise LLMResponseError("解析结果为空")
        except Exception as e:
            logger.error(f"计算查询嵌入失败: {str(e)}", exc_info=True)
            if isinstance(e, LLMResponseError):
                raise
            raise LLMResponseError(f"计算查询嵌入失败: {str(e)}")


# ==================== LLM工厂类 ====================
class LLMFactory:
    """LLM工厂类"""
    _llm_classes = {
        "qwen": QwenLLM,
    }
    
    @classmethod
    def create_llm(
        cls,
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseChatModel:
        """创建LLM实例"""
        provider = provider.lower()
        if provider not in cls._llm_classes:
            raise LLMConfigError(f"不支持的LLM提供商: {provider}")
        
        try:
            llm_class = cls._llm_classes[provider]
            return llm_class(model_name=model_name, api_key=api_key, **kwargs)
        except Exception as e:
            raise LLMConfigError(f"创建{provider} LLM实例失败: {str(e)}") from e


# ==================== 全局实例管理 ====================
_llm_instances: Dict[str, BaseChatModel] = {}
_embedding_instances: Dict[str, Embeddings] = {}

def get_llm(
    provider: str = None,
    model_name: str = None,
    api_key: Optional[str] = None,
    **kwargs
) -> BaseChatModel:
    """获取LLM实例"""
    if provider is None:
        provider = Config.DEFAULT_LLM_PROVIDER
    if model_name is None:
        provider_config = MODEL_CONFIGS.get(provider, {})
        model_name = provider_config.get("chat_model", "qwen-turbo-latest")
    
    instance_key = f"{provider}_{model_name}"
    
    if instance_key in _llm_instances:
        return _llm_instances[instance_key]
    
    llm = LLMFactory.create_llm(provider, model_name, api_key, **kwargs)
    _llm_instances[instance_key] = llm
    return llm


def get_chat_model(**kwargs) -> BaseChatModel:
    """获取聊天模型实例"""
    provider = Config.DEFAULT_LLM_PROVIDER
    provider_config = MODEL_CONFIGS.get(provider, {})
    model_name = provider_config.get("chat_model", "qwen-turbo-latest")
    
    api_key = provider_config.get("api_key", "")
    if not api_key:
        env_key = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(env_key, "")
    
    return get_llm(provider, model_name, api_key, **kwargs)


def get_embedding_model(**kwargs) -> Embeddings:
    """获取嵌入模型实例"""
    provider = Config.DEFAULT_LLM_PROVIDER
    provider_config = MODEL_CONFIGS.get(provider, {})
    model_name = provider_config.get("embedding_model", "text-embedding-v1")
    
    api_key = provider_config.get("api_key", "")
    if not api_key:
        env_key = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(env_key, "")
    
    instance_key = f"embedding_{provider}_{model_name}"
    
    if instance_key in _embedding_instances:
        return _embedding_instances[instance_key]
        
    embedding_model = QwenEmbeddings(model_name=model_name, api_key=api_key, **kwargs)
    _embedding_instances[instance_key] = embedding_model
    return embedding_model


# ==================== 便捷调用函数 ====================
def generate_text(prompt: str, **kwargs) -> str:
    """文本生成快捷函数"""
    llm = get_llm(**kwargs)
    return llm.generate(prompt, **kwargs)


def chat(messages: List[Dict[str, str]], **kwargs) -> str:
    """对话快捷函数"""
    llm = get_llm(**kwargs)
    return llm.chat(messages, **kwargs)
