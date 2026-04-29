"""
SQL Agent模块
提供SQL查询和数据库交互功能
"""
import os
import sys
import logging
import sqlite3
from typing import Dict, List, Optional, Any

# 导入LangChain相关模块
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.language_models.llms import BaseLLM as LangChainBaseLLM

# 导入配置
from .config import Config

# 根据LangChain版本选择正确的导入路径
try:
    # 尝试从新版本导入
    from langchain_community.agent_toolkits import create_sql_agent
except ImportError:
    try:
        # 尝试从旧版本导入
        from langchain.agents import create_sql_agent
    except ImportError:
        # 如果都失败，尝试从agent_toolkits导入
        from langchain.agents.agent_toolkits import create_sql_agent

# 初始化日志
logger = logging.getLogger(__name__)


def get_db_path(db_name: str = "school.db") -> str:
    """
    获取数据库文件的完整路径
    
    Args:
        db_name: 数据库文件名
        
    Returns:
        数据库文件的完整路径
    """
    return os.path.join(Config.COURSE_DATA_DIR, db_name)


def get_db_connection(db_path: str):
    """获取启用了外键约束的数据库连接"""
    conn = sqlite3.connect(db_path)
    # 启用外键约束
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


class SQLAgent:
    """SQL Agent类，封装SQL查询和数据库交互功能"""
    
    def __init__(
        self,
        db_name: str = "school.db",
        db_path: Optional[str] = None,
        llm: Optional[LangChainBaseLLM] = None,
        verbose: bool = True,
        init_tables: bool = True,
        skip_llm: bool = False
    ):
        """
        初始化SQL Agent
        
        Args:
            db_name: 数据库文件名（默认为"school.db"）
            db_path: 数据库文件完整路径（如果提供，将覆盖db_name）
            llm: LLM实例（可选，不提供则使用默认LLM）
            verbose: 是否显示详细日志
            init_tables: 是否初始化基础表（学生、教师、课程）
            skip_llm: 是否跳过LLM初始化（用于测试）
        """
        # 确定数据库路径
        if db_path is None:
            self.db_path = get_db_path(db_name)
        else:
            self.db_path = db_path
            
        self.verbose = verbose
        self.skip_llm = skip_llm
        
        # 初始化LLM（如果需要）
        if not self.skip_llm:
            if llm is None:
                logger.info("使用默认LLM初始化SQL Agent")
                try:
                    from llms import get_chat_model
                    self.llm = get_chat_model()
                except ImportError:
                    logger.warning("无法导入get_chat_model函数，跳过LLM初始化")
                    self.skip_llm = True
                    self.llm = None
            else:
                self.llm = llm
                logger.info("使用传入的LLM初始化SQL Agent")
            
            # 验证LLM类型（如果有LLM）
            if self.llm is not None and not isinstance(self.llm, LangChainBaseLLM):
                raise TypeError("LLM必须是LangChain的BaseLLM实例")
        else:
            self.llm = None
            logger.info("跳过LLM初始化")
        
        # 初始化数据库和Agent
        self._initialize_db()
        
        # 如果需要，初始化基础表
        if init_tables:
            self._init_database_tables()
        
        # 只有在有LLM的情况下才初始化Agent
        if self.llm is not None:
            self._initialize_agent()
    
    def _initialize_db(self):
        """初始化数据库连接"""
        try:
            # 检查数据库文件是否存在，不存在则创建
            if not os.path.exists(self.db_path):
                logger.info(f"创建新的数据库文件: {self.db_path}")
                # 创建空数据库文件
                conn = sqlite3.connect(self.db_path)
                conn.close()
            
            # 创建数据库连接
            self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
            logger.info(f"数据库连接成功: {self.db_path}")
            
        except Exception as e:
            logger.error(f"初始化数据库失败: {str(e)}")
            raise RuntimeError(f"无法初始化数据库: {str(e)}") from e
    
    def _init_database_tables(self):
        """初始化数据库基础表（学生、教师、课程）"""
        try:
            # 使用sqlite3直接连接数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建学生表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                gender TEXT CHECK(gender IN ('男', '女', '其他')),
                age INTEGER,
                grade TEXT,
                class TEXT,
                email TEXT,
                phone TEXT,
                address TEXT,
                enrollment_date DATE DEFAULT CURRENT_DATE
            )
            """)
            
            # 创建教师表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS teachers (
                teacher_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                gender TEXT CHECK(gender IN ('男', '女', '其他')),
                age INTEGER,
                department TEXT,
                title TEXT,
                email TEXT,
                phone TEXT,
                hire_date DATE DEFAULT CURRENT_DATE
            )
            """)
            
            # 创建课程表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS courses (
                course_id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_name TEXT NOT NULL,
                course_code TEXT UNIQUE NOT NULL,
                credits REAL NOT NULL,
                hours INTEGER NOT NULL,
                description TEXT,
                department TEXT,
                teacher_id INTEGER,
                FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id)
            )
            """)
            
            # 创建学生选课表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS enrollments (
                enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                course_id INTEGER NOT NULL,
                enrollment_date DATE DEFAULT CURRENT_DATE,
                grade REAL,
                semester TEXT,
                FOREIGN KEY (student_id) REFERENCES students(student_id),
                FOREIGN KEY (course_id) REFERENCES courses(course_id),
                UNIQUE(student_id, course_id, semester)
            )
            """)
            
            # 提交更改并关闭连接
            conn.commit()
            conn.close()
            
            logger.info("数据库基础表初始化完成")
            
        except Exception as e:
            logger.error(f"初始化数据库表失败: {str(e)}")
            raise RuntimeError(f"无法初始化数据库表: {str(e)}") from e
    
    def _initialize_agent(self):
        """初始化SQL Agent"""
        try:
            # 创建工具包
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            
            # 创建Agent
            self.agent = create_sql_agent(
                llm=self.llm,
                toolkit=toolkit,
                verbose=self.verbose,
                agent_type="tool-calling"
            )
            
            logger.info("SQL Agent初始化完成")
            
        except Exception as e:
            logger.error(f"初始化SQL Agent失败: {str(e)}")
            raise RuntimeError(f"无法初始化SQL Agent: {str(e)}") from e
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        执行SQL查询
        
        Args:
            question: 自然语言问题
            
        Returns:
            包含查询结果的字典
        """
        if self.llm is None:
            logger.warning("未初始化LLM，无法执行自然语言查询")
            return {
                "question": question,
                "answer": "未初始化LLM，无法执行自然语言查询",
                "success": False
            }
            
        try:
            logger.info(f"执行SQL查询: {question}")
            
            # 使用正确的输入格式调用Agent
            result = self.agent.invoke({
                "input": question,
                "chat_history": []
            })
            
            # 解析返回结果
            if isinstance(result, dict):
                # 提取最终答案
                output = result.get("output", str(result))
                
                # 尝试提取中间步骤（SQL查询和结果）
                intermediate_steps = result.get("intermediate_steps", [])
                
                return {
                    "question": question,
                    "answer": output,
                    "intermediate_steps": intermediate_steps,
                    "success": True
                }
            else:
                # 如果返回的不是字典，转换为字符串
                return {
                    "question": question,
                    "answer": str(result),
                    "success": True
                }
                
        except Exception as e:
            logger.error(f"查询失败: {str(e)}")
            return {
                "question": question,
                "answer": f"查询失败: {str(e)}",
                "success": False
            }
    
    def add_sample_data(self):
        """添加示例数据到数据库"""
        try:
            # 使用sqlite3直接连接数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 检查是否已有数据
            cursor.execute("SELECT COUNT(*) FROM students")
            student_count = cursor.fetchone()[0]
            
            if student_count > 0:
                logger.info("数据库中已有数据，跳过示例数据添加")
                conn.close()
                return
            
            # 添加示例教师数据
            teachers_data = [
                ("张教授", "男", 45, "计算机科学", "教授", "zhang@example.com", "13800138001", "2010-09-01"),
                ("李副教授", "女", 38, "数学", "副教授", "li@example.com", "13800138002", "2015-09-01"),
                ("王讲师", "男", 32, "物理", "讲师", "wang@example.com", "13800138003", "2018-09-01")
            ]
            
            cursor.executemany("""
            INSERT INTO teachers (name, gender, age, department, title, email, phone, hire_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, teachers_data)
            
            # 添加示例学生数据
            students_data = [
                ("张三", "男", 20, "大二", "计算机1班", "zhangsan@example.com", "13900139001", "北京市海淀区", "2022-09-01"),
                ("李四", "女", 19, "大一", "数学1班", "lisi@example.com", "13900139002", "北京市朝阳区", "2023-09-01"),
                ("王五", "男", 21, "大三", "物理1班", "wangwu@example.com", "13900139003", "北京市西城区", "2021-09-01")
            ]
            
            cursor.executemany("""
            INSERT INTO students (name, gender, age, grade, class, email, phone, address, enrollment_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, students_data)
            
            # 添加示例课程数据
            courses_data = [
                ("数据结构", "CS101", 3.0, 48, "介绍基本数据结构和算法", "计算机科学", 1),
                ("高等数学", "MATH101", 4.0, 64, "微积分和线性代数基础", "数学", 2),
                ("大学物理", "PHYS101", 3.5, 56, "经典力学和电磁学", "物理", 3)
            ]
            
            cursor.executemany("""
            INSERT INTO courses (course_name, course_code, credits, hours, description, department, teacher_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, courses_data)
            
            # 添加示例选课数据
            enrollments_data = [
                (1, 1, "2023-09-01", 85.5, "2023-2024秋季"),
                (1, 2, "2023-09-01", 90.0, "2023-2024秋季"),
                (2, 2, "2023-09-01", 88.0, "2023-2024秋季"),
                (3, 3, "2023-09-01", 92.5, "2023-2024秋季")
            ]
            
            cursor.executemany("""
            INSERT INTO enrollments (student_id, course_id, enrollment_date, grade, semester)
            VALUES (?, ?, ?, ?, ?)
            """, enrollments_data)
            
            # 提交更改并关闭连接
            conn.commit()
            conn.close()
            
            logger.info("示例数据添加完成")
            
        except Exception as e:
            logger.error(f"添加示例数据失败: {str(e)}")
            raise RuntimeError(f"无法添加示例数据: {str(e)}") from e
    
    def test_database_tables(self):
        """测试数据库表是否正常创建和初始化"""
        print("\n===== 测试数据库表初始化 =====")
        
        try:
            # 使用启用了外键约束的连接创建表
            conn = get_db_connection(self.db_path)
            cursor = conn.cursor()
            
            # 检查表是否存在
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [table[0] for table in cursor.fetchall()]
            
            expected_tables = ['students', 'teachers', 'courses', 'enrollments']
            
            # 检查所有期望的表是否存在
            for table in expected_tables:
                if table in tables:
                    print(f"✓ 表 {table} 存在")
                else:
                    print(f"✗ 表 {table} 不存在")
                    conn.close()
                    return False
            
            # 检查数据是否正确插入
            cursor.execute("SELECT COUNT(*) FROM students")
            student_count = cursor.fetchone()[0]
            print(f"✓ 学生表中有 {student_count} 条记录")
            
            cursor.execute("SELECT COUNT(*) FROM teachers")
            teacher_count = cursor.fetchone()[0]
            print(f"✓ 教师表中有 {teacher_count} 条记录")
            
            cursor.execute("SELECT COUNT(*) FROM courses")
            course_count = cursor.fetchone()[0]
            print(f"✓ 课程表中有 {course_count} 条记录")
            
            cursor.execute("SELECT COUNT(*) FROM enrollments")
            enrollment_count = cursor.fetchone()[0]
            print(f"✓ 选课表中有 {enrollment_count} 条记录")
            
            conn.close()
            print("✓ 数据库表初始化测试通过")
            return True
        except Exception as e:
            print(f"✗ 数据库表初始化测试失败: {str(e)}")
            return False
    
    def test_sql_queries(self):
        """测试SQL查询功能"""
        print("\n===== 测试SQL查询功能 =====")
        
        try:
            # 使用启用了外键约束的连接创建表和数据
            conn = get_db_connection(self.db_path)
            cursor = conn.cursor()
            
            # 测试查询列表
            test_queries = [
                ("查询所有学生的信息", "SELECT * FROM students"),
                ("查询所有教师的信息", "SELECT * FROM teachers"),
                ("查询所有课程的信息", "SELECT * FROM courses"),
                ("查询所有选课记录", "SELECT * FROM enrollments"),
                ("查询选修了'数据结构'课程的学生", """
                    SELECT s.* FROM students s
                    JOIN enrollments e ON s.student_id = e.student_id
                    JOIN courses c ON e.course_id = c.course_id
                    WHERE c.course_name = '数据结构'
                """),
                ("查询张教授教授的课程", """
                    SELECT c.* FROM courses c
                    JOIN teachers t ON c.teacher_id = t.teacher_id
                    WHERE t.name = '张教授'
                """),
                ("查询张三选修的所有课程及成绩", """
                    SELECT c.*, e.grade FROM courses c
                    JOIN enrollments e ON c.course_id = e.course_id
                    JOIN students s ON e.student_id = s.student_id
                    WHERE s.name = '张三'
                """),
                ("查询计算机科学系的所有课程", """
                    SELECT * FROM courses WHERE department = '计算机科学'
                """),
                ("查询选修课程数量最多的学生", """
                    SELECT s.*, COUNT(e.course_id) as course_count FROM students s
                    JOIN enrollments e ON s.student_id = e.student_id
                    GROUP BY s.student_id
                    ORDER BY course_count DESC
                    LIMIT 1
                """),
                ("查询平均成绩最高的课程", """
                    SELECT c.*, AVG(e.grade) as avg_grade FROM courses c
                    JOIN enrollments e ON c.course_id = e.course_id
                    GROUP BY c.course_id
                    ORDER BY avg_grade DESC
                    LIMIT 1
                """)
            ]
            
            # 执行测试查询
            for query_name, query_sql in test_queries:
                print(f"\n测试查询: {query_name}")
                cursor.execute(query_sql)
                results = cursor.fetchall()
                print(f"✓ 查询成功，返回 {len(results)} 条记录")
                if results:
                    print("示例结果:", results[0])
            
            conn.close()
            print("\n✓ 所有SQL查询测试完成")
            return True
        except Exception as e:
            print(f"✗ SQL查询测试失败: {str(e)}")
            return False
    
    def test_database_integrity(self):
        """测试数据库完整性"""
        print("\n===== 测试数据库完整性 =====")
        
        try:
            # 使用启用了外键约束的连接创建表和数据
            conn = get_db_connection(self.db_path)
            cursor = conn.cursor()
            
            # 检查外键约束是否生效
            # 尝试插入一个无效的外键引用
            try:
                cursor.execute("INSERT INTO enrollments (student_id, course_id, semester) VALUES (999, 999, '2023-2024')")
                conn.commit()
                print("✗ 外键约束未生效，允许插入无效引用")
                conn.close()
                return False
            except sqlite3.IntegrityError:
                print("✓ 外键约束生效，拒绝插入无效引用")
            
            # 检查唯一约束是否生效
            # 尝试插入重复的课程代码
            try:
                cursor.execute("INSERT INTO courses (course_name, course_code, credits, hours) VALUES ('测试课程', 'CS101', 3.0, 48)")
                conn.commit()
                print("✗ 唯一约束未生效，允许插入重复课程代码")
                conn.close()
                return False
            except sqlite3.IntegrityError:
                print("✓ 唯一约束生效，拒绝插入重复课程代码")
            
            conn.close()
            print("\n✓ 数据库完整性测试通过")
            return True
        except Exception as e:
            print(f"✗ 数据库完整性测试失败: {str(e)}")
            return False
    
    def run_tests(self):
        """运行所有测试"""
        print("\n========================================")
        print("开始运行SQL Agent和数据库表测试")
        print("========================================")
        
        test_results = []
        
        # 运行各项测试
        test_results.append(("数据库表初始化", self.test_database_tables()))
        test_results.append(("SQL查询功能", self.test_sql_queries()))
        test_results.append(("数据库完整性", self.test_database_integrity()))
        
        # 打印测试结果摘要
        print("\n========================================")
        print("测试结果摘要")
        print("========================================")
        
        for test_name, result in test_results:
            status = "通过" if result else "失败"
            print(f"{test_name}: {status}")
        
        # 检查是否所有测试都通过
        all_passed = all(result for _, result in test_results)
        
        if all_passed:
            print("\n✓ 所有测试通过！数据库表功能正常。")
        else:
            print("\n✗ 部分测试失败，请检查错误信息。")
        
        return all_passed


# ==================== 便捷函数 ====================
def get_sql_agent(
    db_name: str = "school.db",
    db_path: Optional[str] = None,
    llm: Optional[LangChainBaseLLM] = None,
    verbose: bool = True,
    init_tables: bool = True,
    skip_llm: bool = False
) -> SQLAgent:
    """
    获取SQL Agent实例
    
    Args:
        db_name: 数据库文件名（默认为"school.db"）
        db_path: 数据库文件完整路径（如果提供，将覆盖db_name）
        llm: LLM实例（可选）
        verbose: 是否显示详细日志
        init_tables: 是否初始化基础表（学生、教师、课程）
        skip_llm: 是否跳过LLM初始化（用于测试）
        
    Returns:
        SQL Agent实例
    """
    return SQLAgent(db_name, db_path, llm, verbose, init_tables, skip_llm)


def query_sql_agent(
    question: str,
    db_name: str = "school.db",
    db_path: Optional[str] = None,
    llm: Optional[LangChainBaseLLM] = None,
    verbose: bool = True,
    init_tables: bool = True,
    skip_llm: bool = False
) -> Dict[str, Any]:
    """
    快捷查询函数
    
    Args:
        question: 自然语言问题
        db_name: 数据库文件名（默认为"school.db"）
        db_path: 数据库文件完整路径（如果提供，将覆盖db_name）
        llm: LLM实例（可选）
        verbose: 是否显示详细日志
        init_tables: 是否初始化基础表（学生、教师、课程）
        skip_llm: 是否跳过LLM初始化（用于测试）
        
    Returns:
        查询结果字典
    """
    agent = get_sql_agent(db_name, db_path, llm, verbose, init_tables, skip_llm)
    return agent.query(question)


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建临时数据库文件
    db_path = get_db_path("test_school.db")
    
    # 如果文件已存在，先删除
    if os.path.exists(db_path):
        os.remove(db_path)
    
    try:
        # 创建SQL Agent实例，跳过LLM初始化（因为没有API密钥）
        agent = SQLAgent(db_name="test_school.db", skip_llm=True, verbose=True, init_tables=True)
        
        # 添加示例数据
        agent.add_sample_data()
        
        # 运行测试
        success = agent.run_tests()
        
        # 根据测试结果设置退出码
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 清理测试数据库
        if os.path.exists(db_path):
            os.remove(db_path)
