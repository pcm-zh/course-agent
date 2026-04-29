"""
会话管理模块
处理所有与会话相关的逻辑，包括创建、切换、删除会话等
"""

import logging
import re
import uuid
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Callable, Tuple

# 配置日志
from .logger import LoggerManager
from .config import Config

# 创建当前模块的日志实例（命名为模块名）
logger = LoggerManager.get_logger(__name__)

class SessionManager:
    def __init__(self, get_sessions: Callable, create_session: Callable, 
                 get_chat_history: Callable, delete_session: Callable):
        """
        初始化会话管理器
        
        Args:
            get_sessions: 获取所有会话的函数
            create_session: 创建新会话的函数
            get_chat_history: 获取聊天历史的函数
            delete_session: 删除会话的函数
        """
        self.get_sessions = get_sessions
        self.create_session = create_session
        self.get_chat_history = get_chat_history
        self.delete_session_func = delete_session  
        self.current_thread_id = None
        self.current_session_name = None
        self.session_list = []  # 会话历史列表：每个元素是 {"name": 会话名, "create_time": 创建时间}
        
        # 初始化时加载最新会话
        self._load_latest_session()
    
    def _load_latest_session(self):
        """加载最新的会话"""
        try:
            # 1. 尝试获取会话列表
            try:
                self.session_list = self.get_sessions()
            except Exception as db_error:
                # 记录详细的堆栈信息，以便排查
                logger.error(f"读取数据库时发生严重错误: {str(db_error)}", exc_info=True)
                # 不要在这里创建临时会话，而是抛出异常，让上层知道数据库有问题
                raise Exception(f"数据库读取失败: {str(db_error)}")
            
            # 2. 检查列表是否为空
            if not self.session_list:
                logger.info("数据库读取成功，但当前没有会话。正在创建默认会话...")
                try:
                    # 创建默认会话
                    create_success = self.create_session(Config.DEFAULT_SESSION_NAME)
                    if not create_success:
                        raise Exception("创建默认会话函数返回 False")
                    
                    # 重新获取会话列表
                    self.session_list = self.get_sessions()
                    
                    if not self.session_list:
                        raise Exception("创建默认会话后，会话列表仍为空")
                except Exception as create_error:
                    logger.error(f"创建默认会话失败: {create_error}", exc_info=True)
                    raise Exception(f"创建默认会话失败: {str(create_error)}")
            
            # 3. 如果有会话，进行排序和设置
            if self.session_list:
                # 确保列表按时间倒序排列（最新的在最前）
                self.session_list.sort(key=lambda x: x["create_time"], reverse=True)
                self.current_session_name = self.session_list[0]["name"]
                # 设置 current_thread_id 为当前会话名称
                self.current_thread_id = self.current_session_name
                logger.info(f"初始化会话历史成功，当前会话: {self.current_session_name}")
            else:
                # 如果列表依然为空（创建默认会话失败），则抛出异常
                # 只有在这里抛出异常，外层的 except 才会创建临时会话
                raise Exception("数据库为空且无法创建默认会话")

        except Exception as e:
            # 只有在数据库连接失败或严重错误时才进入这里
            # 如果仅仅是 get_sessions() 返回了空列表（因为没数据），上面的逻辑会处理，不会进入这里
            logger.error(f"加载会话列表发生严重错误，启用临时会话模式: {str(e)}", exc_info=True)
            
            # 只有在万不得已的情况下才创建临时会话
            # 并且记录警告日志
            logger.warning("数据库初始化异常，创建临时会话以维持运行（数据可能不同步）")
            default_session = {
                "name": Config.TEMP_SESSION_NAME, 
                "create_time": datetime.now()
            }
            self.session_list = [default_session]
            self.current_session_name = Config.TEMP_SESSION_NAME
            self.current_thread_id = Config.TEMP_SESSION_NAME
    
    def get_current_thread_id(self) -> Optional[str]:
        """获取当前会话的线程ID"""
        return self.current_thread_id
    
    def get_current_session_name(self) -> Optional[str]:
        """获取当前会话的名称"""
        return self.current_session_name
    
    def get_session_list(self) -> List[Dict]:
        """获取会话列表"""
        return self.session_list
    
    def switch_session(self, session_name: str) -> Tuple[List[Dict], str]:
        """
        切换到指定会话
        
        Args:
            session_name: 要切换到的会话名称
            
        Returns:
            tuple: (聊天历史, 会话列表HTML)
        """
        try:
            if not session_name:
                return [{"role": "assistant", "content": "请选择一个有效的会话"}], self._update_session_list()
            
            # 更新当前会话
            self.current_session_name = session_name
            # 更新 current_thread_id 为当前会话名称
            self.current_thread_id = session_name
            logger.info(f"切换到会话: {session_name}")
            
            # 获取会话历史
            chat_history = self.get_chat_history(session_name)
            if not chat_history:
                chat_history = [{"role": "assistant", "content": f"✅ 已为你切换至：{session_name}。\n现在可以继续提问相关课程问题。"}]
            
            # 更新会话列表
            session_list_html = self._update_session_list()
            
            return chat_history, session_list_html
        except Exception as e:
            error_msg = f"切换会话失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return [{"role": "assistant", "content": f"❌ {error_msg}"}], self._update_session_list()
    
    def new_session(self) -> Tuple[List[Dict], str]:
        """
        创建新会话
        
        Returns:
            tuple: (聊天历史, 会话列表HTML)
        """
        try:
            # 修正日期格式化字符串
            today_date_str = datetime.now().strftime("%Y-%m-%d")
            today_sessions = [s for s in self.session_list if s["create_time"].strftime("%Y-%m-%d") == today_date_str]
            
            if not today_sessions:
                new_index = 1
            else:
                indices = []
                
                prefix = Config.NEW_SESSION_PREFIX
                for s in today_sessions:
                    # 正则匹配 "会话 数字" 或 "Session 数字"
                    match = re.search(rf'{prefix}\s*(\d+)', s["name"])
                    if match:
                        indices.append(int(match.group(1))) 
                
                new_index = max(indices) + 1 if indices else 1
            
            # 时间格式化字符串
            timestamp = datetime.now().strftime("%H%M%S")
            
            new_session_name = f"{Config.NEW_SESSION_PREFIX} {new_index}_{timestamp}"
            
            if not self._is_session_name_unique(new_session_name):
                # 如果已存在，添加随机后缀
                new_session_name = f"{Config.NEW_SESSION_PREFIX} {new_index}_{timestamp}_{uuid.uuid4().hex[:4]}" 
            
            # 1. 先写入数据库
            db_success = self.create_session(new_session_name)
            
            if not db_success:
                logger.error(f"数据库创建会话失败: {new_session_name}")
                # 尝试创建默认会话
                logger.info("尝试创建默认会话...")
                
                default_name = f"{Config.DEFAULT_SESSION_NAME}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                db_success = self.create_session(default_name)
                if not db_success:
                    raise Exception("数据库创建会话失败，无法创建默认会话")
                new_session_name = default_name
            
            # 2. 数据库写入成功后，刷新内存列表 
            self.refresh_session_list()
            
            # 3. 设置当前会话
            self.current_session_name = new_session_name
            self.current_thread_id = new_session_name
            
            welcome_msg = [{"role": "assistant", "content": "🎉 已创建新会话，你好！很高兴为你服务\n\n请输入你的课程相关问题，我会尽力解答！"}]
            
            logger.info(f"创建新会话成功: {new_session_name}")
            return welcome_msg, self._update_session_list()
        except Exception as e:
            error_msg = f"创建新会话失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return [{"role": "assistant", "content": f"❌ {error_msg}"}], self._update_session_list()
    
    def delete_session(self, session_name: str) -> Tuple[List[Dict], str]:
        """
        删除指定会话
        
        Args:
            session_name: 要删除的会话名称
            
        Returns:
            tuple: (聊天历史, 会话列表HTML)
        """
        try:
            if not session_name:
                return [{"role": "assistant", "content": "请选择一个有效的会话"}], self._update_session_list()
            
            # 1. 删除数据库中的会话
            db_success = self.delete_session_func(session_name)
            
            if db_success:
                # 2. 数据库删除成功后，刷新内存列表 
                self.refresh_session_list()
                
                # 3. 如果删除的是当前会话，切换到最新的会话
                if session_name == self.current_session_name:
                    if self.session_list:
                        self.current_session_name = self.session_list[0]["name"]
                        self.current_thread_id = self.current_session_name
                        chat_history = self.get_chat_history(self.current_session_name)
                        if not chat_history:
                            chat_history = [{"role": "assistant", "content": f"✅ 已切换至：{self.current_session_name}"}]
                    else:
                        self.current_session_name = ""
                        self.current_thread_id = ""
                        chat_history = [{"role": "assistant", "content": "当前没有会话，请点击'新建会话'开始。"}]
                else:
                    chat_history = self.get_chat_history(self.current_session_name)
                    if not chat_history:
                        chat_history = [{"role": "assistant", "content": f"当前会话：{self.current_session_name}"}]
                
                logger.info(f"删除会话成功: {session_name}")
                return chat_history, self._update_session_list()
            else:
                logger.error(f"删除会话失败: {session_name}")
                error_msg = [{"role": "assistant", "content": f"❌ 删除会话失败，请重试。"}]
                return error_msg, self._update_session_list()
        except Exception as e:
            error_msg = f"删除会话失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return [{"role": "assistant", "content": f"❌ {error_msg}"}], self._update_session_list()
    
    def _is_session_name_unique(self, session_name: str) -> bool:
        """检查会话名是否唯一"""
        return not any(s["name"] == session_name for s in self.session_list)
    
    def _update_session_list(self) -> str:
        """
        更新会话列表HTML
        
        Returns:
            str: 会话列表的HTML字符串
        """
        try:
            session_list_html = ""
            
            grouped_sessions = {
                Config.SESSION_GROUP_TODAY: [],
                Config.SESSION_GROUP_YESTERDAY: [],
                Config.SESSION_GROUP_DAY_BEFORE: [],
                Config.SESSION_GROUP_OLDER: []
            }
            
            today = date.today()
            yesterday = today - timedelta(days=1)
            day_before_yesterday = today - timedelta(days=2)
            
            for sess in self.session_list:
                sess_name = sess["name"]
                create_time = sess["create_time"]
                create_date = create_time.date()
                
                active_class = "background-color: #4285f4; color: white;" if sess_name == self.current_session_name else ""
                
                if create_date == today:
                    grouped_sessions[Config.SESSION_GROUP_TODAY].append((sess_name, active_class))
                elif create_date == yesterday:
                    grouped_sessions[Config.SESSION_GROUP_YESTERDAY].append((sess_name, active_class))
                elif create_date == day_before_yesterday:
                    grouped_sessions[Config.SESSION_GROUP_DAY_BEFORE].append((sess_name, active_class))
                else:
                    grouped_sessions[Config.SESSION_GROUP_OLDER].append((sess_name, active_class))
            
            def get_session_index(name):
                """获取会话索引"""
                prefix = Config.NEW_SESSION_PREFIX
                match = re.search(rf'{prefix}\s*(\d+)', name)
                return int(match.group(1)) if match else 0
            
            group_order = [
                Config.SESSION_GROUP_TODAY, 
                Config.SESSION_GROUP_YESTERDAY, 
                Config.SESSION_GROUP_DAY_BEFORE, 
                Config.SESSION_GROUP_OLDER
            ]
            
            for group_name in group_order:
                sessions_in_group = grouped_sessions.get(group_name, [])
                
                if not sessions_in_group:
                    continue
                
                session_list_html += f"""
                <div style="margin: 0;">
                    <h4 style="margin: 12px 0 6px 0; color: #666; font-size: 14px;">{group_name}</h4>
                """
                
                for sess_name, active_class in sessions_in_group:
                    # 正则表达式转义字符
                    display_name = re.sub(r'_\d{6}$', '', sess_name)
                    
                    svg_icon = """
                    <svg t="1775209082201" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4815" width="32" height="32"><path d="M960.167183 449.688839c0-17.253966-13.980409-31.233352-31.234375-31.233352S897.699456 432.434873 897.699456 449.688839c0 177.968297-172.802648 322.749752-385.217479 322.749752-0.548492 0-1.059122 0.13303-1.600451 0.161682-6.865361-0.231267-13.842263 1.647523-19.761066 6.081509L374.512672 866.139669l0-83.289961c0-1.542122-0.237407-3.018753-0.453325-4.499477 2.406816-14.398942-5.573949-28.961612-19.739577-34.328853-137.929396-52.209082-227.055273-167.739329-227.055273-294.332539 0-17.253966-13.980409-31.233352-31.233352s-31.233352 13.980409-31.233352 31.233352c0 146.375765 96.360651 279.546787 247.248174 344.541057l0 134.378523c0 11.834536 6.680143 22.642696 17.264199 27.939329 4.422729 2.206248 9.201569 3.294023 13.970176 3.294023 6.628978 0 13.227256-2.115174 18.738783-6.243191l158.490687-118.868271C763.656577 831.018777 960.167183 659.787928 960.167183 449.688839z" fill="#1afa29" p-id="4816"></path><path d="M103.473653 374.938368c16.186657 5.876847 34.121122-2.53166 39.987736-18.758226 48.802496-134.971018 200.438055-229.241055 368.766808-229.241055 169.051207 0 320.879148 94.789876 369.224226 230.501768 4.554736 12.780071 16.582677 20.761859 29.424146 20.761859 3.477194 0 7.015787-0.589424 10.482749-1.819438 16.247033-5.78475 24.727171-23.658839 18.941398-39.906895C882.347816 173.778882 710.317765 64.47136 512.228197 64.47136c-197.224874 0-369.021611 108.698654-427.513794 270.480294C78.848813 351.167988 87.246063 369.071753 103.473653 374.938368z" fill="#1afa29" p-id="4817"></path><path d="M390.596999 300.54503c-17.253966 0-31.233352 13.980409-31.233352 31.233352l0 30.470989c0 17.253966 13.980409 31.233352 31.233352 31.233352s31.234375-13.980409 31.234375-31.233352l0-30.470989C421.830351 314.524416 407.850965 300.54503 390.596999 300.54503z" fill="#1afa29" p-id="4818"></path><path d="M634.366955 300.54503c-17.253966 0-31.234375 13.980409-31.234375 31.233352l0 30.470989c0 17.253966 13.980409 31.233352 31.234375 31.233352s31.233352-13.980409 31.233352-31.233352l0-30.470989C665.60133 314.524416 651.620921 300.54503 634.366955 300.54503z" fill="#1afa29" p-id="4819"></path><path d="M589.507258 441.99153c-14.305821-9.405207-33.593096-5.38873-43.159986 8.764618-0.132006 0.193405-13.614066 19.754926-34.172287 19.754926-19.989263 0-32.423457-18.098193-33.267685-19.36914-9.160637-14.417361-28.254507-18.809391-42.834574-9.770528-14.650675 9.109472-19.155269 28.367071-10.055007 43.017746 11.214413 18.047028 41.970904 48.589648 86.157265 48.589648 43.963281 0 75.105558-30.318516 86.574774-48.223305C607.970772 470.358601 603.78238 451.396737 589.507258 441.99153z" fill="#1afa29" p-id="4820"></path></svg>
                    """
                    
                    svg_delete = f"""
                    <svg onclick="event.stopPropagation(); triggerDeleteSession('{sess_name}')" class="delete-icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="1801" width="32" height="32"><path d="M519.620465 0c-103.924093 0-188.511256 82.467721-192.083349 185.820279H85.015814A48.91386 48.91386 0 0 0 36.101953 234.686512a48.91386 48.91386 0 0 0 48.913861 48.866232h54.010046V831.345116c0 102.852465 69.822512 186.844279 155.909954 186.844279h439.200744c86.087442 0 155.909953-83.491721 155.909954-186.844279V284.100465h48.91386a48.91386 48.91386 0 0 0 48.913861-48.890046 48.91386 48.91386 0 0 0-48.913861-48.866233h-227.756651A191.559442 191.559442 0 0 0 519.620465 0z m-107.234232 177.080558c3.548279-49.771163 46.627721-88.540279 99.851907-88.540279 53.224186 0 96.327442 38.745302 99.351813 88.540279h-199.20372z m-111.997024 752.044651c-30.981953 0-65.083535-39.15014-65.083535-95.041488V287.744h575.488v546.839814c0 55.915163-34.077767 95.041488-65.059721 95.041488H300.389209v-0.500093z" fill="#D81E06" p-id="1802"></path><path d="M368.116093 796.814884c24.361674 0 44.27014-21.670698 44.27014-48.818605v-278.623256c0-27.147907-19.908465-48.818605-44.27014-48.818604-24.33786 0-44.27014 21.670698-44.27014 48.818604v278.623256c0 27.147907 19.360744 48.818605 44.293954 48.818605z m154.933581 0c24.361674 0 44.293953-21.670698 44.293954-48.818605v-278.623256c0-27.147907-19.932279-48.818605-44.293954-48.818604-24.33786 0-44.27014 21.670698-44.270139 48.818604v278.623256c0 27.147907 19.932279 48.818605 44.293953 48.818605z m132.810419 0c24.33786 0 44.27014-21.670698 44.27014-48.818605v-278.623256c0-27.147907-19.932279-48.818605-44.27014-48.818604s-44.27014 21.670698-44.27014 48.818604v278.623256c0 27.147907 19.360744 48.818605 44.27014 48.818605z" fill="#D81E06" p-id="1803"></path></svg>
                    """
                    
                    session_list_html += f"""
                    <div class="session-item-wrapper">
                        <button class="session-btn" 
                                onclick="triggerSwitchSession('{sess_name}')" 
                                style="{active_class}">
                            {svg_icon}
                            {display_name}
                        </button>
                        {svg_delete}
                    </div>
                    """
                    
                session_list_html += "</div>"
            
            session_list_html += "</div>"
            return session_list_html
        except Exception as e:
            error_msg = f"更新会话列表失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"<div class='error'>{error_msg}</div>"
    
    def refresh_session_list(self) -> None:
        """刷新会话列表（从数据库重新加载）"""
        try:
            # 重新从数据库获取
            self.session_list = self.get_sessions()
            # 确保按时间倒序排列
            self.session_list.sort(key=lambda x: x["create_time"], reverse=True)
            logger.info(f"刷新会话列表成功，共 {len(self.session_list)} 个会话")
        except Exception as e:
            logger.error(f"刷新会话列表失败: {e}", exc_info=True)
            raise
    
    def get_session_by_name(self, session_name: str) -> Optional[Dict]:
        """根据名称获取会话"""
        for sess in self.session_list:
            if sess["name"] == session_name:
                return sess
        return None
    
    def rename_session(self, session_name: str, new_name: str, update_session_func: Callable) -> bool:
        """
        重命名会话
        
        Args:
            session_name: 原会话名称
            new_name: 新会话名称
            update_session_func: 更新会话的函数
            
        Returns:
            bool: 是否重命名成功
        """
        # 验证会话名称
        if not session_name or not isinstance(session_name, str):
            logger.error("无效的会话名称")
            return False
        
        if not new_name or not isinstance(new_name, str):
            logger.error("无效的新会话名称")
            return False
        
        # 检查会话是否存在
        if not self.get_session_by_name(session_name):
            logger.error(f"会话不存在: {session_name}")
            return False
        
        try:
            # 检查新名称是否已存在
            if not self._is_session_name_unique(new_name):
                logger.error(f"会话名已存在: {new_name}")
                return False
            
            # 更新数据库
            if not update_session_func(session_name, title=new_name):
                logger.error(f"更新数据库失败: {session_name} -> {new_name}")
                return False
            
            # 更新全局列表
            for sess in self.session_list:
                if sess["name"] == session_name:
                    sess["name"] = new_name
                    # 如果重命名的是当前活动会话，更新current_thread_id
                    if session_name == self.current_session_name:
                        self.current_session_name = new_name
                        self.current_thread_id = new_name
                    break
            
            logger.info(f"重命名会话成功: {session_name} -> {new_name}")
            return True
        except Exception as e:
            logger.error(f"重命名会话失败: {e}", exc_info=True)
            return False
