import pathlib
import textwrap
import google.generativeai as genai
import google.generativeai as genai

# [1] 빈칸을 작성하시오.
# API 키
GOOGLE_API_KEY = "AIzaSyDGzJas5k3AHSLbtGpMXZ0FXeG1ARBK0pw"
genai.configure(api_key=GOOGLE_API_KEY)

# 모델 초기화
def LLM(text):
  model = genai.GenerativeModel('gemini-pro')
  
  while True:
    user_input = text
    if user_input=="q":
      break
    else:
      """
      [Example]
      User: I want to change the top of the woman to a blue sweatshirt and the bottom to a black skirt.
      You: {"top":["blue","sweatshirt"],"bottom":["black","skirt"]}
      """
      # [2] 빈칸을 작성하시오.
      # 예시와 같이 성능 향상을 위해 프롬프트 튜닝을 진행
      instruction = ""
      prompt = ""

      # [3] 빈칸을 작성하시오.
      # 전체 프롬프트 생성 (instruction 포함)
      full_prompt = 

      # [4] 빈칸을 작성하시오.
      # 모델 호출
      response = 

      # 응답 출력
      return response
