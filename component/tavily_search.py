"""
Tavily 网络检索工具
利用Tavily API 进行实时网络检索，获取最新资讯。
核心功能：
1. 封装Tavily API客户端，实现单例模式管理
2. 提供灵活的网络检索接口，支持多参数配置
3. 格式化检索结果，便于Agent使用和展示
4. 提供结果提取和摘要生成的辅助功能
5. 包含完善的异常处理和配置检查
"""
# 导入操作系统模块，用于环境变量处理
import os
# 导入类型注解，用于参数/返回值类型提示
from typing import Optional, Dict, List, Any
# 导入Tavily客户端（需安装：pip install tavily-python）
from tavily import TavilyClient

# 导入项目内部模块
from .logger import LoggerManager  # 日志管理器
from .config import Config         # 项目配置

# 创建当前模块的日志实例（命名为模块名）
logger = LoggerManager.get_logger(__name__)

# ==================== 全局客户端管理 ====================
# 全局Tavily客户端实例（单例模式），初始为None
_client: Optional[TavilyClient] = None

def get_tavily_client() -> TavilyClient:
    """
    获取Tavily客户端实例（单例模式）
    确保全局只有一个客户端实例，避免重复创建连接
    
    Returns:
        TavilyClient: 初始化完成的Tavily客户端
        
    Raises:
        ValueError: API Key未配置时抛出
        RuntimeError: 客户端初始化失败时抛出
    """
    # 声明使用全局变量
    global _client
    
    # 单例检查：如果实例不存在则创建
    if _client is None:
        # 从配置获取API Key
        api_key = Config.TAVILY_API_KEY
        
        # 检查API Key是否配置
        if not api_key:
            logger.warning("TAVILY_API_KEY未配置，将返回模拟结果")
            raise ValueError("TAVILY_API_KEY未配置，请设置环境变量")
        
        try:
            # 创建Tavily客户端实例
            _client = TavilyClient(api_key=api_key)
            logger.info("Tavily客户端初始化成功")
            
        except Exception as e:
            # 记录初始化失败错误并抛出运行时异常
            logger.error(f"Tavily客户端初始化失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"无法初始化Tavily客户端: {str(e)}") from e
    
    # 返回单例实例
    return _client

def tavily_search(
    query: str, 
    max_results: int = None,
    search_depth: str = None,
    include_domains: List[str] = None,
    exclude_domains: List[str] = None,
    days: int = None,
    max_tokens: int = None
) -> str:
    """
    执行网络检索（核心功能）
    封装Tavily API的搜索功能，提供参数默认值和结果格式化
    
    Args:
        query: 检索关键词（必选）
        max_results: 返回结果数（默认使用配置值TAVILY_MAX_RESULTS）
        search_depth: 搜索深度，可选 "basic"（基础）或 "advanced"（高级）（默认配置值）
        include_domains: 限制搜索结果的域名列表（如["gov.cn"]）
        exclude_domains: 排除搜索结果的域名列表
        days: 限制搜索结果的时间范围（天数），如7表示仅返回近7天的结果
        max_tokens: 限制返回内容的最大字符数（防止内容过长）

    Returns:
        str: 格式化的检索结果（包含AI总结和搜索结果列表）
    """
    # 前置检查：API Key未配置时返回模拟结果
    if not Config.TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY未配置，返回模拟结果")
        return f"【模拟结果】关于「{query}」的最新资讯（需配置Tavily API Key）"

    try:
        # ==================== 参数处理（使用配置默认值） ====================
        # 使用配置值作为默认值，允许传入参数覆盖
        max_results = max_results or Config.TAVILY_MAX_RESULTS
        search_depth = search_depth or Config.TAVILY_SEARCH_DEPTH
        include_domains = include_domains or Config.TAVILY_INCLUDE_DOMAINS
        exclude_domains = exclude_domains or Config.TAVILY_EXCLUDE_DOMAINS
        
        # 获取客户端实例
        client = get_tavily_client()
        # 记录检索日志
        logger.info(f"Tavily检索：{query}（结果数：{max_results}，深度：{search_depth}）")

        # ==================== 构建搜索参数 ====================
        search_params = {
            "query": query,                      # 检索关键词
            "max_results": max_results,          # 返回结果数量
            "search_depth": search_depth,        # 搜索深度
            "include_answer": True,              # 包含AI总结答案
            "include_raw_content": False,        # 不包含原始页面内容
            "include_images": False,             # 不包含图片
            "include_image_descriptions": False, # 不包含图片描述
        }
        
        # 添加可选参数（仅当参数有值时）
        if include_domains:
            search_params["include_domains"] = include_domains  # 白名单域名
        if exclude_domains:
            search_params["exclude_domains"] = exclude_domains  # 黑名单域名
        if days:
            search_params["days"] = days                        # 时间范围

        # ==================== 执行搜索 ====================
        response = client.search(**search_params)

        # ==================== 格式化结果 ====================
        results = []
        
        # 1. 添加AI总结（如果有）
        if response.get("answer"):
            results.append(f"AI总结：{response['answer']}\n")

        # 2. 添加每条搜索结果
        for i, res in enumerate(response.get("results", []), 1):  # 从1开始编号
            title = res.get("title", "无标题")       # 结果标题
            url = res.get("url", "")                 # 结果链接
            content = res.get("content", "").strip() # 结果摘要
            
            # 限制内容长度（防止内容过长）
            if max_tokens:
                content = content[:max_tokens]      # 使用指定长度
            else:
                content = content[:500]             # 默认限制500字符
            
            # 格式化单条结果
            results.append(f"{i}. 标题：{title}\n   链接：{url}\n   摘要：{content}...\n")

        # 处理无结果情况
        if not results:
            return "未找到相关结果"
        
        # 拼接所有结果并返回
        return "".join(results)

    except ValueError as e:
        # 处理已知的配置错误（如API Key问题）
        error_msg = f"检索配置错误：{str(e)}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        # 处理其他意外错误（网络问题、API返回异常等）
        error_msg = f"检索失败：{str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

def extract_search_results(response: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    从Tavily API原始响应中提取结构化搜索结果
    简化结果处理，只保留核心字段
    
    Args:
        response: Tavily API返回的原始响应字典
        
    Returns:
        List[Dict[str, str]]: 结构化结果列表，每个字典包含title/url/content/score
    """
    results = []
    # 遍历原始结果列表
    for res in response.get("results", []):
        # 提取核心字段（提供默认值避免KeyError）
        result = {
            "title": res.get("title", ""),    # 标题
            "url": res.get("url", ""),        # 链接
            "content": res.get("content", ""),# 内容摘要
            "score": res.get("score", 0)      # 相关性得分
        }
        results.append(result)
    return results

def get_search_summary(query: str, max_results: int = 3) -> str:
    """
    获取搜索结果的简要摘要（轻量版搜索）
    仅返回AI总结或第一条结果的简短摘要，适合快速获取核心信息
    
    Args:
        query: 搜索查询关键词
        max_results: 返回结果数（默认3条）
        
    Returns:
        str: 简洁的搜索摘要（最多200字符）
    """
    # 前置检查：API Key未配置时返回模拟结果
    if not Config.TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY未配置，返回模拟摘要")
        return f"【模拟摘要】关于「{query}」的简要信息（需配置Tavily API Key）"
    
    try:
        # 获取客户端实例
        client = get_tavily_client()
        # 执行轻量级搜索（基础深度，仅获取AI答案）
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",       # 基础搜索（速度更快）
            include_answer=True,        # 启用AI总结
            include_raw_content=False   # 不返回原始内容
        )
        
        # 优先返回AI生成的总结答案
        if response.get("answer"):
            return response["answer"]
        
        # 备用方案：返回第一条结果的简短摘要
        results = response.get("results", [])
        if results:
            return f"根据搜索结果，{results[0].get('content', '')[:200]}"
        
        # 无结果时的提示
        return "未找到相关信息"
    
    except Exception as e:
        # 异常处理：记录错误并返回友好提示
        logger.error(f"获取搜索摘要时出错: {str(e)}", exc_info=True)
        return f"获取搜索摘要失败: {str(e)}"

if __name__ == "__main__":
    """
    模块独立运行时的测试入口
    验证Tavily搜索功能是否正常工作
    """
    # 测试基本搜索功能
    print("=== 基本搜索测试 ===")
    print(tavily_search("2024年秋季学期课程安排", max_results=2))
