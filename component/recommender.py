import logging
import re
from typing import List, Dict, Optional, Any

# 辅助函数：标准化消息对象
def _normalize_message(msg: Any) -> Dict:
    """将消息对象标准化为字典，兼容 sqlite3.Row 和 dict"""
    if isinstance(msg, dict):
        return msg
    return {key: msg[key] for key in msg.keys()}

class QuestionRecommender:
    # 修改常量：要求推荐5个以上
    DEFAULT_MAX_TURNS = 5
    MAX_RECOMMENDATIONS = 5 
    
    # 匹配中文实体，长度放宽以匹配更多词组
    ENTITY_PATTERN = re.compile(r'[\u4e00-\u9fa5]{2,6}')
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 扩充规则库，确保每个类别至少有5个以上的问题
        self.rules = {
            "价格": [
                "这个服务的收费标准是什么？", 
                "有免费试用吗？", 
                "如何充值？", 
                "充值有优惠吗？", 
                "发票怎么开？",
                "支持哪些支付方式？"
            ],
            "教程": [
                "有使用手册吗？", 
                "在哪里可以看视频教程？", 
                "新手入门指南", 
                "有没有进阶教程？", 
                "API 文档在哪里？",
                "常见问题解答在哪里看？"
            ],
            "账号": [
                "如何修改密码？", 
                "如何绑定手机号？", 
                "忘记密码怎么办？", 
                "如何注销账号？", 
                "可以修改注册邮箱吗？",
                "如何查看账号安全等级？"
            ],
            "错误": [
                "遇到报错怎么办？", 
                "连接超时如何解决？", 
                "显示 404 怎么办？", 
                "上传失败怎么处理？", 
                "数据加载不出来怎么办？",
                "如何查看错误日志？"
            ],
            "上传": [
                "支持什么格式的文件？", 
                "文件大小有限制吗？", 
                "上传失败怎么办？", 
                "支持批量上传吗？", 
                "上传后可以修改吗？",
                "如何下载已上传的文件？"
            ],
            # 默认推荐也扩充到6个
            "默认": [
                "你能做什么？", 
                "介绍一下这个功能", 
                "联系人工客服", 
                "系统有哪些最新功能？", 
                "如何反馈意见？",
                "有相关的使用案例吗？"
            ]
        }
        
        # 增加追问模板，丰富多样性
        self.follow_up_templates = [
            "关于{topic}，我想了解更多",
            "能详细解释一下{topic}吗？",
            "{topic}的具体步骤是什么？",
            "除了{topic}，还有其他方法吗？",
            "如果{topic}失败了怎么办？",
            "关于{topic}的注意事项是什么？"
        ]

    def _get_last_message_by_role(self, context_window: List[Dict], role: str) -> str:
        """从上下文中获取特定角色的最后一条消息内容"""
        for msg in reversed(context_window):
            if msg.get("role") == role:
                return msg.get("content", "")
        return ""

    def _match_keywords(self, text: str) -> Optional[List[str]]:
        """基于关键词匹配推荐问题"""
        for keyword, questions in self.rules.items():
            if keyword == "默认":
                continue
            if keyword in text:
                # 返回所有匹配的问题，后续再截取
                return questions
        return None

    def _generate_follow_up(self, text: str) -> List[str]:
        """基于文本生成追问，尝试生成更多样化的结果"""
        # 提取中文实体
        entities = self.ENTITY_PATTERN.findall(text)
        if not entities:
            return []
        
        # 去重并取前3个实体，以便生成更多追问
        unique_entities = list(dict.fromkeys(entities))[:3]
        
        recommendations = []
        # 遍历实体和模板，生成组合
        for entity in unique_entities:
            for template in self.follow_up_templates:
                recommendations.append(template.format(topic=entity))
                # 如果已经够多了，提前停止
                if len(recommendations) >= self.MAX_RECOMMENDATIONS:
                    return recommendations
        
        return recommendations

    def predict(self, thread_id: str, context_window: Optional[List[Dict]] = None) -> List[str]:
        """
        预测用户可能想问的问题
        
        Args:
            thread_id: 当前会话ID
            context_window: 当前上下文（如果为None，会自动获取）
            
        Returns:
            推荐问题列表 (长度为 MAX_RECOMMENDATIONS)
        """
        try:
            # 1. 获取并标准化上下文
            if context_window is None:
                from component.memory_sqlite import get_context_window
                raw_context = get_context_window(thread_id, max_turns=self.DEFAULT_MAX_TURNS)
                context_window = [_normalize_message(msg) for msg in raw_context]
            
            if not context_window:
                # 如果没有上下文，随机打乱默认推荐并返回前5个
                import random
                defaults = self.rules["默认"].copy()
                random.shuffle(defaults)
                return defaults[:self.MAX_RECOMMENDATIONS]
            
            # 2. 获取最后一条用户消息
            last_user_msg = self._get_last_message_by_role(context_window, "user")
            
            # 3. 尝试基于关键词匹配
            keyword_matches = self._match_keywords(last_user_msg)
            if keyword_matches:
                # 随机打乱匹配结果，避免每次推荐顺序都一样
                import random
                random.shuffle(keyword_matches)
                return keyword_matches[:self.MAX_RECOMMENDATIONS]
            
            # 4. 尝试基于助手回复生成追问
            last_assistant_msg = self._get_last_message_by_role(context_window, "assistant")
            follow_ups = self._generate_follow_up(last_assistant_msg)
            
            # 5. 混合策略：如果追问不足5个，补充默认推荐
            if follow_ups:
                if len(follow_ups) < self.MAX_RECOMMENDATIONS:
                    # 从默认推荐中补充，避免重复
                    defaults = self.rules["默认"]
                    # 简单的去重逻辑（实际应用中可能需要更复杂的语义去重）
                    existing_set = set(follow_ups)
                    for q in defaults:
                        if len(follow_ups) >= self.MAX_RECOMMENDATIONS:
                            break
                        if q not in existing_set:
                            follow_ups.append(q)
                return follow_ups[:self.MAX_RECOMMENDATIONS]
            
            # 6. 兜底返回默认推荐
            import random
            defaults = self.rules["默认"].copy()
            random.shuffle(defaults)
            return defaults[:self.MAX_RECOMMENDATIONS]
            
        except Exception as e:
            self.logger.error(f"生成推荐问题时出错: {e}", exc_info=True)
            return self.rules["默认"][:self.MAX_RECOMMENDATIONS]

# 初始化推荐器实例
recommender = QuestionRecommender()
