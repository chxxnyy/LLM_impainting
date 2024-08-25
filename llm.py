import pathlib
import textwrap
import google.generativeai as genai
import google.generativeai as genai

# [1] 빈칸을 작성하시오.
# API 키
GOOGLE_API_KEY = "AIzaSyA7dnePb8hyIQCiDNPF0Diq2FPofuCt6jI"
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
      instruction = (
                "Extract the clothing items and their colors from the following description. "
                "Return the result as a Python dictionary with two keys: 'top' and 'bottom'. "
                "For each key, provide a list that contains the color as the first element and the item as the second element. "
                "If there are additional details, include them as further elements in the list starting from the third position. "
                "Ensure that each key's value is a single list, not a nested list."
      )
      prompt = user_input

      # [3] 빈칸을 작성하시오.
      # 전체 프롬프트 생성 (instruction 포함)
      full_prompt = f"{instruction}\n{prompt}"

      # [4] 빈칸을 작성하시오.
      # 모델 호출
      response = model.generate_content(full_prompt)

      # 응답 출력
      return response
