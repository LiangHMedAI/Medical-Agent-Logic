# Medical Safety Audit Kernel v1.0
# Developed by LiangHMedAI

def audit_medication(drug, symptom, history):
    """
    业务逻辑：阿司匹林止咳判定及禁忌审计
    """
    if drug == "阿司匹林":
        if "咳嗽" in symptom:
            return "Warning: 阿司匹林不止咳，请针对性选药。"
        if "胃" in history:
            return "Alert: 胃粘膜受损风险，禁用。"
    return "Audit Passed."

# 模拟运行
print(audit_medication("阿司匹林", "我咳嗽了", "我有胃病"))
