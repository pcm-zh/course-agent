"""
日志管理模块
提供并发安全的日志记录功能，支持日志文件按大小轮转，全局单例管理
核心特性：
1. 单例模式确保全局日志配置唯一
2. 支持并发安全的日志文件轮转（优先使用ConcurrentRotatingFileHandler）
3. 兼容配置模块导入失败的降级处理
4. 支持控制台和文件双输出，可配置日志级别
"""
# 导入标准日志模块
import logging
# 导入操作系统模块，用于路径处理
import os
# 导入系统模块，用于控制台输出
import sys
# 导入类型注解，用于参数类型提示
from typing import Optional

# ==================== 日志处理器导入与降级处理 ====================
try:
    # 优先导入并发安全的日志轮转处理器（需安装：pip install concurrent-log-handler）
    from concurrent_log_handler import ConcurrentRotatingFileHandler
    CONCURRENT_HANDLER_AVAILABLE = True  # 标记并发处理器可用
except ImportError:
    # 如果并发处理器不可用，降级使用标准库的轮转处理器
    from logging.handlers import RotatingFileHandler
    CONCURRENT_HANDLER_AVAILABLE = False  # 标记并发处理器不可用

# ==================== 配置导入与降级处理 ====================
try:
    # 尝试从当前包导入项目统一配置
    from .config import Config
except ImportError:
    # 配置导入失败时，创建降级配置类（保证模块可独立运行）
    class Config:
        LOG_FILE = "app.log"                  # 默认日志文件路径
        MAX_BYTES = 10 * 1024 * 1024         # 默认日志文件最大大小（10MB）
        BACKUP_COUNT = 5                     # 默认备份文件数量
        LOG_LEVEL = logging.INFO             # 默认日志级别
        LOG_TO_CONSOLE = True                # 默认输出到控制台


# ==================== 核心日志管理器类（单例模式） ====================
class LoggerManager:
    """
    日志管理器类（单例模式）
    负责全局日志配置的初始化和管理，确保整个应用使用统一的日志配置
    特性：
    1. 单例模式：全局仅一个实例，避免重复配置
    2. 并发安全：优先使用ConcurrentRotatingFileHandler处理多线程日志写入
    3. 灵活扩展：支持获取命名logger，便于模块级日志管理
    """
    # 单例实例存储（类级变量）
    _instance = None           # 存储唯一的LoggerManager实例
    _initialized = False       # 标记是否已完成初始化
    _logger = None             # 存储根logger实例
    _loggers = {}              # 存储命名logger实例的字典（模块名 -> logger实例）

    def __new__(cls, *args, **kwargs):
        """
        重写__new__方法实现单例模式
        确保无论创建多少次LoggerManager实例，都返回同一个对象
        
        Returns:
            LoggerManager的唯一实例
        """
        # 如果实例尚未创建，调用父类方法创建新实例
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        # 返回已创建的实例（或新创建的实例）
        return cls._instance

    def __init__(self):
        """
        初始化日志管理器（构造函数）
        仅在首次创建实例时执行初始化逻辑（通过_initialized标记控制）
        """
        # 防止重复初始化（单例模式下__init__可能被多次调用）
        if not LoggerManager._initialized:
            # 执行核心初始化逻辑
            self._init_logger()
            # 标记为已初始化
            LoggerManager._initialized = True

    def _init_logger(self):
        """
        核心初始化方法：配置日志格式、处理器、级别等
        私有方法，仅在实例首次初始化时调用
        """
        # ==================== 确保日志目录存在 ====================
        # 提取日志文件所在目录
        log_dir = os.path.dirname(Config.LOG_FILE)
        # 如果目录非空且不存在，则创建（exist_ok=True避免重复创建报错）
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # ==================== 配置日志格式 ====================
        # 日志格式字符串：时间 + 级别 + 日志器名 + 文件:行号 + 消息
        fmt = "%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s"
        # 时间格式：年-月-日 时:分:秒
        datefmt = "%Y-%m-%d %H:%M:%S"
        # 创建格式器对象
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

        # ==================== 创建文件日志处理器 ====================
        if CONCURRENT_HANDLER_AVAILABLE:
            # 优先使用并发安全的轮转处理器（适合多线程/多进程场景）
            handler = ConcurrentRotatingFileHandler(
                Config.LOG_FILE,               # 日志文件路径
                maxBytes=Config.MAX_BYTES,     # 单个文件最大字节数
                backupCount=Config.BACKUP_COUNT,  # 备份文件数量
                encoding="utf-8"               # 编码格式（确保中文正常显示）
            )
            handler_type = "ConcurrentRotatingFileHandler"  # 记录处理器类型
        else:
            # 降级使用标准轮转处理器（单线程场景）
            handler = RotatingFileHandler(
                Config.LOG_FILE,
                maxBytes=Config.MAX_BYTES,
                backupCount=Config.BACKUP_COUNT,
                encoding="utf-8"
            )
            handler_type = "RotatingFileHandler"

        # 为处理器设置日志格式
        handler.setFormatter(formatter)

        # ==================== 配置根Logger ====================
        # 获取根logger实例（全局默认logger）
        LoggerManager._logger = logging.getLogger()
        
        # 获取日志级别（使用getattr兼容配置缺失的情况，默认INFO）
        log_level = getattr(Config, 'LOG_LEVEL', logging.INFO)
        # 设置根logger的日志级别（低于该级别的日志不会被记录）
        LoggerManager._logger.setLevel(log_level)
        
        # 清空已有处理器（避免重复添加导致日志重复输出）
        LoggerManager._logger.handlers.clear()
        # 添加文件处理器
        LoggerManager._logger.addHandler(handler)

        # ==================== 配置控制台输出（可选） ====================
        # 获取是否输出到控制台的配置（兼容配置缺失）
        log_to_console = getattr(Config, 'LOG_TO_CONSOLE', True)
        if log_to_console:
            # 创建控制台处理器（输出到标准输出stdout，而非stderr）
            console_handler = logging.StreamHandler(sys.stdout)
            # 设置控制台日志格式
            console_handler.setFormatter(formatter)
            # 添加控制台处理器
            LoggerManager._logger.addHandler(console_handler)

        # ==================== 记录初始化日志 ====================
        LoggerManager._logger.info(f"日志管理器初始化完成，使用处理器：{handler_type}")
        LoggerManager._logger.info(f"日志文件：{Config.LOG_FILE}，最大大小：{Config.MAX_BYTES}字节，备份数：{Config.BACKUP_COUNT}")
        LoggerManager._logger.info(f"日志级别：{logging.getLevelName(log_level)}")

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """
        获取logger实例（类方法，支持命名logger）
        命名logger便于区分不同模块的日志，同时共享统一配置
        
        Args:
            name: logger名称（通常为模块名，如__name__），None则返回根logger
            
        Returns:
            配置完成的logging.Logger实例
        """
        # 确保LoggerManager已初始化（如果尚未初始化则自动初始化）
        if not cls._initialized:
            cls()  # 触发初始化
        
        if name:
            # 如果指定了名称，优先从缓存获取
            if name in cls._loggers:
                return cls._loggers[name]
            
            # 缓存未命中：创建新的命名logger
            logger = logging.getLogger(name)
            # 设置日志级别（与根logger保持一致）
            log_level = getattr(Config, 'LOG_LEVEL', logging.INFO)
            logger.setLevel(log_level)
            # 存入缓存
            cls._loggers[name] = logger
            return logger
        
        # 未指定名称：返回根logger
        return cls._logger

    def set_level(self, level: str):
        """
        动态设置日志级别（支持运行时调整）
        
        Args:
            level: 日志级别字符串（DEBUG/INFO/WARNING/ERROR/CRITICAL）
        """
        # 统一转换为大写，避免大小写问题
        level = level.upper()
        # 验证级别有效性
        if level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            # 将字符串级别转换为logging模块的常量
            log_level = getattr(logging, level, logging.INFO)
            # 更新根logger级别
            LoggerManager._logger.setLevel(log_level)
            # 更新所有已创建的命名logger级别（确保全局级别统一）
            for logger in LoggerManager._loggers.values():
                logger.setLevel(log_level)
            # 记录级别变更日志
            LoggerManager._logger.info(f"日志级别已设置为：{level}")
        else:
            # 无效级别时记录警告
            LoggerManager._logger.warning(f"无效的日志级别：{level}，支持的级别：DEBUG/INFO/WARNING/ERROR/CRITICAL")


# ==================== 全局便捷接口 ====================
# 创建全局默认logger实例（模块导入时自动初始化）
logger = LoggerManager.get_logger()

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取logger实例的便捷函数
    简化调用方式，无需直接操作LoggerManager类
    
    Args:
        name: logger名称（可选）
        
    Returns:
        logging.Logger实例
    """
    return LoggerManager.get_logger(name)

def set_log_level(level: str):
    """
    设置日志级别的便捷函数
    简化日志级别调整操作
    
    Args:
        level: 日志级别字符串（如"DEBUG"）
    """
    LoggerManager().set_level(level)