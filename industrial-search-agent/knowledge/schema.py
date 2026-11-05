"""
知识库结构化定义。

两套分类体系:
1. 三层结构(layer):基础层(事实类)/产业层(关系类)/场景层(应用类)
   —— 对应任务规划文档的工业知识结构图,用于知识粒度分层。
2. 八大类(category):对应《产业知识库目录》一~八章,用于检索过滤与统计。

提供 classify_tag(text) 基于关键词规则给 chunk 打标签。
规则会随真实语料补充逐步完善,这是第1周可落地、第4周可量化的基础。
"""
from __future__ import annotations
from enum import Enum
from typing import Optional


# ===== 三层结构:知识粒度分层 =====
class Layer(str, Enum):
    BASIC = "基础层"       # 事实类:零部件定义、机器人分类、基础参数
    INDUSTRY = "产业层"    # 关系类:产业链上下游、企业-产品关系、技术路线
    SCENE = "场景层"       # 应用类:展厅导览、问答、政策解读


# ===== 八大类:对应知识库目录一~八章 =====
class Category(str, Enum):
    POLICY = "一、宏观政策与产业规划"
    CHAIN = "二、机器人与具身智能产业链"
    STATUS = "三、产业发展现状与趋势"
    CHANGZHOU = "四、常州市及武进区产业基础与实践"
    LAYOUT = "五、空间布局与产业生态"
    ENTERPRISE = "六、重点企业与项目"
    SCENARIO = "七、应用场景与融合赋能"
    STRATEGY = "八、发展对策与建议"


# ===== 关键词规则:category 映射 =====
# 顺序敏感,匹配优先级从上到下;关键词尽量选领域专属词以减少误判。
CATEGORY_KEYWORDS = [
    (Category.POLICY, [
        "政府工作报告", "行动计划", "指导意见", "实施意见", "政策", "规划",
        "工信部", "十七部门", "未来产业", "国家战略",
    ]),
    (Category.CHAIN, [
        "减速器", "伺服", "传感器", "控制器", "核心零部件", "产业链",
        "上游", "中游", "下游", "整机", "本体", "具身智能", "大脑", "小脑",
        "柔性电子皮肤", "一体化关节", "电池包",
    ]),
    (Category.STATUS, [
        "市场规模", "安装量", "四大家族", "长三角", "珠三角", "京津冀",
        "发展趋势", "智能化等级", "机遇", "挑战",
    ]),
    (Category.CHANGZHOU, [
        "常州", "武进", "武高新", "高新区", "规上产值", "单项冠军",
        "专精特新", "协作机器人", "节卡", "安川",
    ]),
    (Category.LAYOUT, [
        "一核多基地", "枢纽平台", "三库", "两中心", "一空间", "一专班",
        "空间布局", "核心集聚区", "新北区", "钟楼区", "经开区",
        "创新载体", "产学研", "双高协同",
    ]),
    (Category.ENTERPRISE, [
        "中车戚墅堰", "坤维科技", "祥明智能", "国茂精密", "恒立精密",
        "艾迈斯", "日盈电子", "睿尔曼", "鼎智科技", "江苏雷利", "豪森润博",
        "微亿智造", "库曼智能", "智元机器人", "创造力", "重点企业",
    ]),
    (Category.SCENARIO, [
        "1+X+N", "具身智能+", "应用场景", "工业", "民生", "公共服务",
        "城市安防", "大田作业", "农业", "医疗康复", "深度融合",
    ]),
    (Category.STRATEGY, [
        "错位竞争", "招商引资", "人才培养", "攻关", "对策", "建议",
        "战略引领", "配套服务",
    ]),
]


# ===== 关键词规则:layer 映射 =====
LAYER_KEYWORDS = [
    (Layer.BASIC, [
        "定义", "分类", "分为", "是指", "包括", "由", "构成", "参数",
        "ISO", "国家标准", "成本", "占比",
    ]),
    (Layer.INDUSTRY, [
        "产业链", "上下游", "企业", "龙头", "市占率", "供应链", "技术路线",
        "规模", "产值", "销量", "量产",
    ]),
    (Layer.SCENE, [
        "应用", "场景", "展厅", "导览", "解读", "推广", "部署",
        "验证", "示范",
    ]),
]


def _match(text: str, rules) -> Optional[Enum]:
    """按规则列表顺序匹配,返回首个命中项。"""
    text_lower = text
    for label, keywords in rules:
        for kw in keywords:
            if kw in text_lower:
                return label
    return None


def classify_category(text: str) -> Category:
    """给一段文本打八大类标签,默认归到产业链类。"""
    return _match(text, CATEGORY_KEYWORDS) or Category.CHAIN


def classify_layer(text: str) -> Layer:
    """给一段文本打三层结构标签,默认归到产业层。"""
    return _match(text, LAYER_KEYWORDS) or Layer.INDUSTRY


def classify_tag(text: str) -> dict:
    """一次性返回 category + layer 标签字典。"""
    return {
        "category": classify_category(text).value,
        "layer": classify_layer(text).value,
    }


if __name__ == "__main__":
    # 自测
    samples = [
        "谐波减速器是工业机器人的核心零部件,目前主要被日本企业垄断。",
        "2025年政府工作报告首次将具身智能产业列入国家战略。",
        "常州市具身智能机器人场景体验中心集聚了多家代表性企业展品。",
        "针对常州缺少整机制造企业的问题,建议重点招引工业机器人项目。",
    ]
    for s in samples:
        print(s)
        print("  ->", classify_tag(s))
