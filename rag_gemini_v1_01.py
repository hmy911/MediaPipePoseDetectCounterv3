# pip install google-generativeai
import google.generativeai as genai
import json

# 你的 Google Cloud Platform API
mygoogleapikey = ''  # 替换API 密碼
genai.configure(api_key=mygoogleapikey)
model = genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_response(input, image):
    if input != "":
        response = model.generate_content([input, image])
    else:
        response = model.generate_content(image)
    return response.text

def jsonDetail(jsonfile):
    """读取 JSON 文件并构建提示信息."""
    with open(jsonfile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 从 JSON 文件中提取信息
    myprompt = f"{data}, 請以單日運動量推測每周 提供50字專業性熱量分析,運動及健康建議專業醫師回應 回應樣式:(姓名, 年紀, 青年中年男女)每日的基礎代謝率? 您的運動量每日推估?大卡 加上飲食?大卡 (夠或不夠) ，例如每周至少 (幾次?何種强度的 跑步或yoga 一周內 騎車 跑步幾分鐘? yoga幾分鐘?的組合)運動，每次至少 ? 分鐘。 早晨的營養均衡，但建議增加蛋白質攝取，例如雞蛋的優質蛋白，以提供更持久的能量。請用繁體中文回答"
    return myprompt

#JSON 文件名
# pathjson = "json/user_data.json"# 男性測試資料
pathjson = "json/user_data2.json" # 女性測試資料
response = get_gemini_response(jsonDetail(pathjson),'')
print(response)



