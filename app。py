import gradio as gr
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# --- 1. 加载你训练好的“大脑” ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)  # 必须是4
model.load_state_dict(torch.load('resnet_model_v2_4classes.pth'))  # 确认文件名正确
model.to(device)
model.eval()

class_names = ['aspirin', 'book', 'pen', 'phone']


# --- 2. 定义 Agent 的决策逻辑 ---
def agent_decision(label):
    actions = {
        'aspirin': "💊 发现阿司匹林！请注意：严禁与抗凝药同服！",
        'phone': "📱 识别为手机。已开启智能办公助手，为您拦截骚扰电话。",
        'book': "📚 识别为书籍。正在检索豆瓣评分，建议开启护眼模式。",
        'pen': "✍️ 识别为文具。已为您打开备忘录，随时记录灵感。"
    }
    return actions.get(label, "❓ 未知物体，Agent 正在观察...")


def agent_medical_logic(label, history=""):
    """
    这是梁工的临床经验库，专门负责审计模型识别出的结果
    """
    # 定义我们的专业逻辑
    if label == "aspirin":
        if "胃" in history or "溃疡" in history:
            return "🚨【拦截建议】：识别为阿司匹林，但检测到您有胃病史。该药刺激胃粘膜，极易诱发出血，请禁用！"
        elif "咳嗽" in history:
            return "⚠️【药效提醒】：识别为阿司匹林。注意：阿司匹林为非甾体抗炎药，对止咳无效，请勿误用。"
        else:
            return "✅【建议】：识别为阿司匹林，请严格遵医嘱，餐后服用以减少胃部刺激。"

    # 如果识别出的是布洛芬
    if label == "布洛芬":
        if "胃" in history:
            return "🚨【严重警告】：布洛芬与胃溃疡高度冲突，禁用！"
        return "✅【建议】：识别为布洛芬，用于解热镇痛，注意每日剂量限制。"

    return f"✅ 识别结果为 {label}，建议咨询专业药师获取详细用法。"
# --- 3. 图像处理与预测函数 ---
def predict(img,history):
    if img is None: return "请上传图片", None

    # 预处理 (必须和训练时一致)
    loader = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_tensor = loader(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output[0], dim=0)

    # 获取最高概率的索引
    conf, pred = torch.max(probabilities, 0)
    label = class_names[pred.item()]

    # 获取决策建议
    action = agent_medical_logic(label, history)

    # 返回结果：{类别: 概率}, 决策文本
    res_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    return res_dict, action


# --- 4. 构建 Gradio 界面 ---
with gr.Blocks(theme=gr.themes.Soft(), title="MedVision Agent") as demo:
    gr.Markdown("# 🏥 药学智能 Agent 实验室")
    gr.Markdown("### 上传一张物品照片，Agent 将为您进行决策分析")

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="📸 投放照片")
            # 1. 先定义病史输入框
            history_input = gr.Textbox(label="🧠 请输入您的病史或症状")
            # 2. 再定义按钮（这样下面第 93 行才能识别出 run_btn）
            run_btn = gr.Button("🚀 启动 Agent 决策", variant="primary")

        with gr.Column():
            label_output = gr.Label(label="👁️ 视觉识别分析")
            action_output = gr.Textbox(label="💬 Agent 决策指令", interactive=False)

    # 3. 最后绑定点击事件（位置一定要在 run_btn 定义之后！）
    run_btn.click(
        fn=predict,
        inputs=[img_input, history_input],
        outputs=[label_output, action_output]
    )

# 启动 (生成本地和临时公网链接)
demo.launch(share=True)
