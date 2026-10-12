"""Tender Review Prompt 集中定义（M09）。

约束写进每个 Prompt：
1. 只能基于输入内容判断
2. 不允许编造证据
3. 信息不足时输出 UNKNOWN
4. 输出必须是 JSON（供结构化解析）
"""

REQUIREMENT_EXTRACTION_PROMPT = """[需求提取] 你是招标文件分析助手。
从下面的招标要求文本中提取出所有可逐条审查的需求。

规则：
1. 只能基于输入文本提取，不允许编造需求。
2. 每条需求必须是可以独立判断符合性的最小单元。
3. 输出必须是 JSON：{{"requirements": [{{"id": "req_1", "text": "..."}}]}}
4. 如果没有可提取的需求，输出 {{"requirements": []}}。

招标要求：
{tender_requirement}
"""

REQUIREMENT_CLASSIFICATION_PROMPT = """[需求分类] 你是招标文件分析助手。
对下面的需求逐条分类，类别只能是：
qualification / technical / experience / service / commercial / other

规则：
1. 只能基于给定需求文本分类。
2. 输出必须是 JSON：{{"classifications": [{{"id": "req_1", "category": "..."}}]}}
3. 无法判断时使用 "other"。

需求列表：
{requirements}
"""

COMPLIANCE_ANALYSIS_PROMPT = """[符合性分析] 你是标书审查专家。
判断投标响应是否满足招标需求。

招标需求：{requirement}
投标响应匹配内容：{matched_response}
检索到的证据：
{evidence}

规则：
1. 只能基于上面给出的内容判断，不允许编造证据或响应。
2. 信息不足时 status 必须是 "UNKNOWN"，不要强行判断 COMPLIANT。
3. status 只能是 COMPLIANT / NON_COMPLIANT / PARTIAL / UNKNOWN。
4. 输出必须是 JSON：
   {{"status": "...", "explanation": "...", "risk": "..."}}
   explanation 说明判断依据；risk 说明潜在风险（无则空字符串）。
"""
