"""
工具注册模块
定义Agent可调用的所有工具
核心功能：
1. 封装并注册所有可用工具（课程检索/网络检索/SQL查询/文件解析）
2. 实现工具执行监控（耗时/错误率统计）
3. 提供统一的工具调用接口和结果标准化
4. 定义工具输入参数模型，确保参数合法性
"""
# 导入类型注解，用于参数/返回值类型提示
from typing import List, Dict, Any, Optional, Callable
# 导入装饰器工具，用于保留函数元信息
from functools import wraps
# 导入LangChain工具相关模块
from langchain.tools import tool
from langchain_core.tools import StructuredTool
# 导入Pydantic，用于定义输入参数模型
from pydantic import BaseModel, Field

# 导入项目内部模块
from .logger import LoggerManager       # 日志管理器
from .rag_course import rag_course_query  # 课程RAG检索工具
from .tavily_search import tavily_search    # 网络检索工具
from .sql_agent import query_sql_agent      # SQL查询工具
from .files_parser import download_file_from_minio, extract_text_from_file
from .models import ToolResult              # 工具结果模型

# 创建当前模块的日志实例
logger = LoggerManager.get_logger(__name__)

# ==================== 工具执行统计 ====================
# 全局字典，存储工具执行统计信息（名称: 统计数据）
_tool_execution_stats: Dict[str, Dict[str, Any]] = {}



def track_tool_execution(tool_name: str):
    """
    装饰器工厂函数：用于跟踪工具执行时间和统计信息
    增加更详细的日志记录和性能监控
    
    Args:
        tool_name: 工具名称（用于标识统计数据）
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import time
            start_time = time.time()
            
            # 初始化该工具的统计信息
            if tool_name not in _tool_execution_stats:
                _tool_execution_stats[tool_name] = {
                    "count": 0,
                    "total_time": 0.0,
                    "errors": 0,
                    "max_time": 0.0,  # 最大执行时间
                    "min_time": float('inf'),  # 最小执行时间
                    "last_execution_time": None  # 最后执行时间
                }
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # 更新统计信息
                _tool_execution_stats[tool_name]["count"] += 1
                _tool_execution_stats[tool_name]["total_time"] += execution_time
                _tool_execution_stats[tool_name]["max_time"] = max(
                    _tool_execution_stats[tool_name]["max_time"], 
                    execution_time
                )
                _tool_execution_stats[tool_name]["min_time"] = min(
                    _tool_execution_stats[tool_name]["min_time"], 
                    execution_time
                )
                _tool_execution_stats[tool_name]["last_execution_time"] = execution_time
                
                # 记录详细日志
                logger.info(f"工具 {tool_name} 执行成功，耗时: {execution_time:.2f}秒")
                logger.debug(f"工具参数: {kwargs}")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # 更新错误统计
                _tool_execution_stats[tool_name]["count"] += 1
                _tool_execution_stats[tool_name]["errors"] += 1
                _tool_execution_stats[tool_name]["total_time"] += execution_time
                
                # 记录详细错误日志
                logger.error(f"工具 {tool_name} 执行失败，耗时: {execution_time:.2f}秒，错误: {str(e)}", exc_info=True)
                logger.debug(f"工具参数: {kwargs}")
                
                raise
        
        return wrapper
    return decorator



def get_tool_execution_stats() -> Dict[str, Dict[str, Any]]:
    """
    获取工具执行统计信息（增强版）
    增加更多统计指标：最大/最小执行时间、成功率等
    
    Returns:
        Dict[str, Dict[str, Any]]: 包含各工具详细统计的字典
    """
    stats = {}
    for tool_name, data in _tool_execution_stats.items():
        # 计算衍生指标
        count = data["count"]
        total_time = data["total_time"]
        errors = data["errors"]
        
        avg_time = total_time / count if count > 0 else 0
        error_rate = errors / count if count > 0 else 0
        success_rate = (count - errors) / count if count > 0 else 0
        
        # 构建增强的统计信息
        stats[tool_name] = {
            "count": count,
            "total_time": total_time,
            "average_time": avg_time,
            "max_time": data.get("max_time", 0.0),
            "min_time": data.get("min_time", 0.0) if data.get("min_time") != float('inf') else 0.0,
            "errors": errors,
            "success_count": count - errors,
            "error_rate": error_rate,
            "success_rate": success_rate,
            "last_execution_time": data.get("last_execution_time", None)
        }
    
    return stats


def reset_tool_execution_stats():
    """重置工具执行统计信息（用于测试/监控重置）"""
    global _tool_execution_stats
    # 清空统计字典
    _tool_execution_stats = {}
    logger.info("已重置工具执行统计信息")


def check_tool_health(tool_name: str) -> Dict[str, Any]:
    """
    检查工具健康状态
    
    Args:
        tool_name: 工具名称
        
    Returns:
        Dict[str, Any]: 包含健康状态信息的字典
    """
    if tool_name not in _tool_execution_stats:
        return {
            "tool_name": tool_name,
            "status": "unknown",
            "message": "工具从未执行过"
        }
    
    stats = _tool_execution_stats[tool_name]
    error_rate = stats["errors"] / stats["count"] if stats["count"] > 0 else 0
    avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
    
    # 判断健康状态
    if error_rate > 0.5:
        status = "unhealthy"
        message = f"工具错误率过高 ({error_rate:.1%})"
    elif avg_time > 10.0:
        status = "warning"
        message = f"工具平均执行时间过长 ({avg_time:.2f}秒)"
    else:
        status = "healthy"
        message = "工具运行正常"
    
    return {
        "tool_name": tool_name,
        "status": status,
        "message": message,
        "error_rate": error_rate,
        "average_time": avg_time,
        "total_executions": stats["count"]
    }

def analyze_tool_performance(tool_name: str, window_size: int = 10) -> Dict[str, Any]:
    """
    分析工具性能趋势
    
    Args:
        tool_name: 工具名称
        window_size: 分析窗口大小（最近N次执行）
        
    Returns:
        Dict[str, Any]: 包含性能分析结果的字典
    """
    if tool_name not in _tool_execution_stats:
        return {
            "tool_name": tool_name,
            "status": "no_data",
            "message": "工具从未执行过"
        }
    
    stats = _tool_execution_stats[tool_name]
    total_executions = stats["count"]
    
    if total_executions < window_size:
        return {
            "tool_name": tool_name,
            "status": "insufficient_data",
            "message": f"工具执行次数不足（{total_executions} < {window_size}）",
            "total_executions": total_executions
        }
    
    # 计算性能指标
    avg_time = stats["total_time"] / total_executions
    error_rate = stats["errors"] / total_executions
    
    # 性能评级
    if error_rate > 0.3:
        performance_grade = "D"
    elif avg_time > 5.0:
        performance_grade = "C"
    elif error_rate > 0.1:
        performance_grade = "B"
    else:
        performance_grade = "A"
    
    return {
        "tool_name": tool_name,
        "status": "success",
        "performance_grade": performance_grade,
        "total_executions": total_executions,
        "average_time": avg_time,
        "error_rate": error_rate,
        "success_rate": 1 - error_rate,
        "recommendation": self._get_performance_recommendation(performance_grade, avg_time, error_rate)
    }

def _get_performance_recommendation(grade: str, avg_time: float, error_rate: float) -> str:
    """生成性能优化建议"""
    if grade == "D":
        return "工具错误率过高，建议检查工具实现和错误处理逻辑"
    elif grade == "C":
        return "工具执行时间过长，建议优化算法或增加缓存"
    elif grade == "B":
        return "工具性能一般，建议进一步优化错误处理"
    else:
        return "工具性能良好，继续保持"

# ==================== 工具输入参数模型 ====================
# 使用Pydantic BaseModel定义工具输入参数，确保类型和描述的完整性
# LangChain会自动使用这些模型生成工具调用的参数说明

class RagCourseInput(BaseModel):
    """课程检索工具输入模型（参数校验+描述）"""
    query: str = Field(description="需要检索的课程问题或关键词")


class TavilySearchInput(BaseModel):
    """网络检索工具输入模型"""
    query: str = Field(description="需要检索的问题或关键词")
    max_results: int = Field(default=3, description="返回结果数量，默认为3")


class SqlAgentInput(BaseModel):
    """SQL查询工具输入模型"""
    query: str = Field(description="需要查询的自然语言问题")


class FileParserInput(BaseModel):
    """文件解析工具输入模型"""
    file_path: str = Field(description="需要解析的文件路径")


# ==================== 工具实现函数 ====================
# 定义工具的实际执行逻辑，这些函数将被装饰器包装并注册为工具

@track_tool_execution("rag_course")
def rag_course(query: str) -> str:
    """检索相关课程内容"""
    return rag_course_query(query)


@track_tool_execution("tavily_search")
def tavily_search_tool(query: str, max_results: int = 3) -> str:
    """网络检索"""
    return tavily_search(query, max_results=max_results)


@track_tool_execution("sql_agent")
def sql_agent_tool(query: str) -> str:
    """SQL数据库查询"""
    return query_sql_agent(query)


@track_tool_execution("files_parser")
def files_parser_tool(file_path: str) -> str:
    """解析文件内容"""
    try:
        # 提取文件内容
        content = extract_text_from_file(file_path)
        if content:
            return f"文件内容已成功提取，共 {len(content)} 个字符。"
        else:
            return "无法从文件中提取内容，请检查文件格式是否正确。"
    except Exception as e:
        return f"解析文件时出错: {str(e)}"


# ==================== 工具注册 ====================
# 代码位置：component/tools.py#L180-L220
def get_tools() -> List[StructuredTool]:
    """
    核心函数：获取Agent可用的工具列表（优化版）
    增强工具描述，添加更多使用场景说明
    
    Returns:
        List[StructuredTool]: LangChain兼容的工具列表
    """
    tools = [
        # 课程检索工具
        StructuredTool.from_function(
            func=rag_course,
            name="rag_course",
            description="必须用于回答所有课程相关问题。当用户询问课程内容、知识点、大纲、习题、作业、考试、成绩等任何与课程相关的问题时，必须使用此工具。参数query为用户的问题。此工具会从向量数据库中检索相关课程资料，并基于检索结果生成回答。",
            args_schema=RagCourseInput
        ),
        # 网络检索工具
        StructuredTool.from_function(
            func=tavily_search_tool,
            name="tavily_search",
            description="进行网络检索获取最新资讯，适用于需要最新信息的问题。例如：最新技术动态、行业新闻、政策变化等。参数query为搜索关键词，max_results为返回结果数量（默认3个）。",
            args_schema=TavilySearchInput
        ),
        # SQL查询工具
        StructuredTool.from_function(
            func=sql_agent_tool,
            name="sql_agent",
            description="使用自然语言查询数据库，适用于数据统计/查询类问题。例如：查询学生成绩、统计课程选课人数、分析学习数据等。参数query为自然语言问题。",
            args_schema=SqlAgentInput
        ),
        # 文件解析工具
        StructuredTool.from_function(
            func=files_parser_tool,
            name="files_parser",
            description="解析文件内容，支持PDF、DOCX、PPTX、TXT、HTML、IPYNB等格式。当用户请求分析特定文件时，必须使用此工具提取文件内容。参数file_path为文件路径。此工具会自动识别文件类型并使用相应的解析器提取文本内容。",
            args_schema=FileParserInput
        )
    ]
    
    logger.info(f"注册工具列表：{[t.name for t in tools]}")
    return tools


def get_tool_by_name(tool_name: str) -> Optional[StructuredTool]:
    """
    根据名称获取工具对象（工具查找辅助函数）
    
    Args:
        tool_name: 工具名称（如"rag_course"）
        
    Returns:
        StructuredTool: 匹配的工具对象，不存在则返回None
    """
    # 获取所有工具
    tools = get_tools()
    # 遍历查找匹配名称的工具
    for tool in tools:
        if tool.name == tool_name:
            return tool
    # 未找到返回None
    return None


def execute_tool(tool_name: str, **kwargs) -> ToolResult:
    """
    统一的工具执行接口（标准化结果）
    调用指定工具，返回包含执行状态的ToolResult对象
    
    Args:
        tool_name: 工具名称
        **kwargs: 工具执行参数（如query="Python课程大纲"）
        
    Returns:
        ToolResult: 标准化的工具执行结果（包含成功状态、结果、耗时、错误信息）
    """
    import time
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 根据名称获取工具
        tool = get_tool_by_name(tool_name)
        if tool is None:
            raise ValueError(f"工具不存在: {tool_name}")
        
        # 执行工具
        result = tool.run(kwargs)
        # 计算执行耗时
        execution_time = time.time() - start_time
        
        # 返回成功的ToolResult
        return ToolResult(
            tool_name=tool_name,
            success=True,
            result=result,
            execution_time=execution_time
        )
        
    except Exception as e:
        # 计算执行耗时（即使出错）
        execution_time = time.time() - start_time
        # 返回失败的ToolResult（包含错误信息）
        return ToolResult(
            tool_name=tool_name,
            success=False,
            result="",
            execution_time=execution_time,
            error=str(e)
        )


if __name__ == "__main__":
    """
    模块独立运行时的测试入口
    验证所有工具是否能正常注册和调用
    """
    # 获取工具列表
    tools = get_tools()
    print(f"已注册工具：{[t.name for t in tools]}")
    
    # 测试工具执行统计
    print("\n=== 工具执行统计测试 ===")
    reset_tool_execution_stats()
    
    # 模拟工具调用
    try:
        # 测试rag_course工具
        print("\n测试 rag_course 工具...")
        result = execute_tool("rag_course", query="数据结构课程")
        print(f"结果: {result}")
        
        # 测试tavily_search工具
        print("\n测试 tavily_search 工具...")
        result = execute_tool("tavily_search", query="Python最新版本")
        print(f"结果: {result}")
        
        # 测试sql_agent工具
        print("\n测试 sql_agent 工具...")
        result = execute_tool("sql_agent", query="学生成绩")
        print(f"结果: {result}")
        
        # 测试files_parser工具
        print("\n测试 files_parser 工具...")
        # 注意：这里需要一个真实的文件路径
        # result = execute_tool("files_parser", file_path="test.pdf")
        # print(f"结果: {result}")
        
    except Exception as e:
        print(f"工具测试失败: {str(e)}")
    
    # 显示工具执行统计
    print("\n=== 工具执行统计 ===")
    stats = get_tool_execution_stats()
    for tool_name, data in stats.items():
        print(f"{tool_name}:")
        print(f"  调用次数: {data['count']}")
        print(f"  总耗时: {data['total_time']:.2f}秒")
        print(f"  平均耗时: {data['average_time']:.2f}秒")
        print(f"  错误次数: {data['errors']}")
        print(f"  错误率: {data['error_rate']:.2%}")
