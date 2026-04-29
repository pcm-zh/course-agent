"""
项目统一配置模块
集中管理所有全局配置项，确保配置的一致性和可维护性
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ==================== 加载环境变量 ====================
# 尝试加载项目根目录或组件目录下的 .env 文件
env_path = Path(__file__).resolve().parent / '.env'
if not env_path.exists():
    env_path = Path(__file__).resolve().parent.parent / '.env'

if env_path.exists():
    load_dotenv(env_path)
else:
    # 如果找不到 .env，也不报错，使用默认值
    pass

# ==================== 基础路径配置 ====================
# 获取项目根目录 (假设该文件位于 component/ 目录下，向上两级是根目录)
BASE_DIR = Path(__file__).resolve().parent.parent

# ==================== 核心配置类 ====================
class Config:
    """
    全局配置类
    所有配置项均通过环境变量获取，同时设置合理的默认值
    确保配置项可通过 .env 文件或系统环境变量灵活配置
    """
    
    # ==================== 基础路径配置 (放在最前面，确保优先定义) ====================
    # 项目根目录
    BASE_DIR = Path(__file__).resolve().parent.parent  # 保持为 Path 对象
    # 数据存储根目录
    # 修改使用 BASE_DIR 的地方
    DATA_DIR = str(BASE_DIR / "data")
    ASSETS_DIR = str(BASE_DIR / "assets")
    
    # 课程数据目录
    COURSE_DATA_DIR = os.getenv("COURSE_DATA_DIR", str(Path(DATA_DIR) / "course"))
    
    # ==================== Gradio/Web 界面配置 ====================
    # Gradio 服务器配置
    GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    
    # FastAPI 后端地址（供前端调用）
    API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    
    # 资源文件路径 (CSS/JS) - 放在 assets 目录下
    ASSETS_DIR = str(BASE_DIR / "assets")
    STYLE_CSS_PATH = str(Path(ASSETS_DIR) / "style.css")
    SCRIPT_JS_PATH = str(Path(ASSETS_DIR) / "script.js")

    # ==================== 日志配置 ====================
    # 日志文件存储路径，默认：data/logs/app.log
    LOG_FILE = os.getenv("LOG_FILE", str(Path(DATA_DIR) / "logs" / "app.log"))
    
    # 文件上传限制
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 32 * 1024 * 1024))  # 默认32MB
    # 支持的文件扩展名
    ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS", "pdf,docx,pptx,txt,html,ipynb").split(",")
    
    # 单个日志文件最大字节数，默认5MB
    MAX_BYTES = int(os.getenv("MAX_BYTES", 5 * 1024 * 1024))
    # 日志文件备份数量
    BACKUP_COUNT = int(os.getenv("BACKUP_COUNT", 3))
    # 日志级别
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    # 是否输出日志到控制台
    LOG_TO_CONSOLE = os.getenv("LOG_TO_CONSOLE", "True").lower() in ("true", "1", "yes")

    # ==================== LLM (大语言模型) 配置 ====================
    DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "qwen")
    LLM_TYPE = os.getenv("LLM_TYPE", "qwen")
    
    # 通义千问配置
    QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
    QWEN_CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL", "qwen-plus")
    QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v1")
    
    # OpenAI配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # LLM通用参数
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2000))
    TOP_P = float(os.getenv("TOP_P", 0.9))
    FREQUENCY_PENALTY = float(os.getenv("FREQUENCY_PENALTY", 0))
    PRESENCE_PENALTY = float(os.getenv("PRESENCE_PENALTY", 0))
    
    # ==================== MinIO 对象存储配置 ====================
    # MinIO服务器配置
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET = os.getenv("MINIO_BUCKET", "course-materials")
    MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() in ("true", "1", "yes")
    MINIO_REGION = os.getenv("MINIO_REGION", "us-east-1")
    
    # MinIO存储路径配置
    MINIO_COURSE_MATERIALS_PATH = os.getenv("MINIO_COURSE_MATERIALS_PATH", "course_materials")
    MINIO_GENERAL_FILES_PATH = os.getenv("MINIO_GENERAL_FILES_PATH", "general_files")

    # ==================== 数据库配置 ====================
    # SQL Agent数据库路径
    SQL_AGENT_DB_PATH = os.getenv("SQL_AGENT_DB_PATH", str(Path(DATA_DIR) / "log_sql" / "course.db"))
    # 聊天历史数据库路径 (存储会话和消息记录)
    CHAT_HISTORY_DB_PATH = os.getenv("CHAT_HISTORY_DB_PATH", str(Path(DATA_DIR) / "log_sql" / "chat_history.db"))
    
    # ==================== 向量库配置 ====================
    # 会话数据库路径
    SESSION_DB_PATH = os.getenv("SESSION_DB_PATH", str(Path(DATA_DIR) / "session.db"))
    # 会话数据目录
    SESSION_DIR = os.getenv("SESSION_DIR", str(Path(DATA_DIR) / "session"))

    # 课程文档目录
    COURSE_DOC_DIR = os.getenv("COURSE_DOC_DIR", str(Path(DATA_DIR) / "course_doc"))
    # 普通文件目录
    GENERAL_FILE_DIR = os.getenv("GENERAL_FILE_DIR", str(Path(DATA_DIR) / "general_files"))
    
    # Chroma向量库持久化目录
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(Path(DATA_DIR) / "chroma_course"))
    # Chroma集合名称
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "course_materials")
    # Chroma距离计算度量方式
    CHROMA_DISTANCE_METRIC = os.getenv("CHROMA_DISTANCE_METRIC", "cosine")
    
    # ==================== 文本分割配置 ====================
    # 文本分割配置
    TEXT_SPLITTER_CHUNK_SIZE = int(os.getenv("TEXT_SPLITTER_CHUNK_SIZE", 500))
    TEXT_SPLITTER_CHUNK_OVERLAP = int(os.getenv("TEXT_SPLITTER_CHUNK_OVERLAP", 50))
    TEXT_SPLITTER_SEPARATORS = os.getenv("TEXT_SPLITTER_SEPARATORS", "\n\n,\n,。,；,，, ,").split(",")

    # ==================== 向量检索配置 ====================
    # 向量检索配置
    VECTOR_SEARCH_TOP_K = int(os.getenv("VECTOR_SEARCH_TOP_K", 3))
    VECTOR_SEARCH_SCORE_THRESHOLD = float(os.getenv("VECTOR_SEARCH_SCORE_THRESHOLD", 0.6))

    # ==================== 课程相关性检查配置 ====================
    # 课程相关性检查配置
    RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", 0.5))
    RELEVANCE_CACHE_ENABLED = os.getenv("RELEVANCE_CACHE_ENABLED", "True").lower() in ("true", "1", "yes")
    RELEVANCE_CACHE_SIZE = int(os.getenv("RELEVANCE_CACHE_SIZE", 128))

    # ==================== 工具配置 ====================
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", 3))
    TAVILY_SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "advanced")
    TAVILY_INCLUDE_DOMAINS = os.getenv("TAVILY_INCLUDE_DOMAINS", "").split(",") if os.getenv("TAVILY_INCLUDE_DOMAINS") else []
    TAVILY_EXCLUDE_DOMAINS = os.getenv("TAVILY_EXCLUDE_DOMAINS", "").split(",") if os.getenv("TAVILY_EXCLUDE_DOMAINS") else []

    # ==================== Agent配置 ====================
    AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", 10))
    AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", 120))
    AGENT_VERBOSE = os.getenv("AGENT_VERBOSE", "False").lower() in ("true", "1", "yes")

    # ==================== 会话管理配置 ====================
    # 默认会话名称
    DEFAULT_SESSION_NAME = os.getenv("DEFAULT_SESSION_NAME", "默认会话")
    
    # 临时会话名称（当数据库不可用时使用）
    TEMP_SESSION_NAME = os.getenv("TEMP_SESSION_NAME", "临时会话")

    # 会话列表分组标签（按时间分组）
    SESSION_GROUP_TODAY = os.getenv("SESSION_GROUP_TODAY", "今天")
    SESSION_GROUP_YESTERDAY = os.getenv("SESSION_GROUP_YESTERDAY", "昨天")
    SESSION_GROUP_DAY_BEFORE = os.getenv("SESSION_GROUP_DAY_BEFORE", "前天")
    SESSION_GROUP_OLDER = os.getenv("SESSION_GROUP_OLDER", "更早")

    # 新建会话时的名称前缀
    NEW_SESSION_PREFIX = os.getenv("NEW_SESSION_PREFIX", "会话")

    # ==================== Agent/交互配置 ====================
    # 默认交互配置
    DEFAULT_THREAD_ID = os.getenv("DEFAULT_THREAD_ID", "1")
    DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "1")
    MAX_TURNS = int(os.getenv("MAX_TURNS", 10)) # 默认最大对话轮次

    # ==================== 终端UI配置 ====================
    # ANSI 颜色代码
    ANSI_HEADER = '\033[95m'
    ANSI_OKBLUE = '\033[94m'
    ANSI_OKCYAN = '\033[96m'
    ANSI_OKGREEN = '\033[92m'
    ANSI_WARNING = '\033[93m'
    ANSI_FAIL = '\033[91m'
    ANSI_END = '\033[0m'
    ANSI_BOLD = '\033[1m'
    
    ANSI_USER = '\033[92m'
    ANSI_AGENT = '\033[94m'
    ANSI_SYSTEM = '\033[93m'
    ANSI_ERROR = '\033[91m'
    ANSI_INFO = '\033[96m'
    ANSI_KNOWLEDGE = '\033[95m'

    # ==================== 缓存配置 ====================
    # 缓存配置
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "True").lower() in ("true", "1", "yes")
    CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))  # 缓存过期时间（秒）

    # ==================== 性能配置 ====================
    # 性能配置
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", 10))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))  # 请求超时时间（秒）

    # ==================== 安全配置 ====================
    # 安全配置
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

    # ==================== 开发/生产环境配置 ====================
    # 开发/生产环境配置
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes") if ENVIRONMENT == "development" else False

    # ==================== 监控配置 ====================
    # 监控配置
    MONITORING_ENABLED = os.getenv("MONITORING_ENABLED", "False").lower() in ("true", "1", "yes")
    MONITORING_ENDPOINT = os.getenv("MONITORING_ENDPOINT", "http://localhost:9090")

    # ==================== 备份配置 ====================
    # 备份配置
    BACKUP_ENABLED = os.getenv("BACKUP_ENABLED", "False").lower() in ("true", "1", "yes")
    BACKUP_DIR = os.getenv("BACKUP_DIR", str(Path(DATA_DIR) / "backups"))
    BACKUP_INTERVAL_HOURS = int(os.getenv("BACKUP_INTERVAL_HOURS", 24))

    # ==================== 文件处理配置 ====================
    # 文件处理配置
    FILE_PROCESSING_THREADS = int(os.getenv("FILE_PROCESSING_THREADS", 4))
    FILE_PROCESSING_TIMEOUT = int(os.getenv("FILE_PROCESSING_TIMEOUT", 300))  # 文件处理超时时间（秒）

    # ==================== 初始化目录 ====================
    @staticmethod
    def init_directories():
        """
        初始化项目所需的所有目录
        确保程序运行前，日志、数据库、向量库等目录已存在
        """
        # 定义需要创建的目录列表
        # 优先使用 Config 类中的配置，如果 Config 中是字符串，转换为 Path 对象处理
        dirs_to_create = [
            Path(Config.DATA_DIR), # 确保本地数据目录存在
            Path(Config.LOG_FILE).parent,          # 日志目录
            Path(Config.CHAT_HISTORY_DB_PATH).parent, # 记忆数据库目录
            Path(Config.SQL_AGENT_DB_PATH).parent, # SQL Agent数据库目录
            Path(Config.SESSION_DIR),              # 会话目录
            Path(Config.COURSE_DOC_DIR),          # 课程文档目录
            Path(Config.GENERAL_FILE_DIR),        # 普通文件目录
            Path(Config.CHROMA_PERSIST_DIR),      # 向量库目录
            Path(Config.COURSE_DATA_DIR),         # 课程数据目录
            BASE_DIR / "logs",                    # 兼容旧版日志目录
            Path(Config.BACKUP_DIR) if Config.BACKUP_ENABLED else None  # 备份目录
        ]
        
        for dir_path in dirs_to_create:
            if dir_path and not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"已创建目录: {dir_path}")

# ==================== 模型配置映射 ====================
MODEL_CONFIGS = {
    "qwen": {
        "api_key": Config.QWEN_API_KEY,
        "chat_model": Config.QWEN_CHAT_MODEL,
        "embedding_model": Config.QWEN_EMBEDDING_MODEL,
        "base_url": "https://dashscope.aliyuncs.com/compatible-model/v1"
    },
    "openai": {
        "api_key": Config.OPENAI_API_KEY,
        "chat_model": Config.OPENAI_CHAT_MODEL,
        "embedding_model": Config.OPENAI_EMBEDDING_MODEL,
        "base_url": "https://api.openai.com/v1"
    }
}

# 导出默认LLM类型
DEFAULT_LLM_TYPE = Config.LLM_TYPE

# ==================== 程序启动时初始化 ====================
# 确保在类定义外部调用初始化方法，这样所有属性都已经定义好了
if __name__ == "__main__":
    Config.init_directories()
