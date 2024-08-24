import os, sys
import argparse
import copy
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
from huggingface_hub import hf_hub_download

def preprop_for_diffusion(image, vis_output_model):
    # 이미지 전처리: (3, 128, 128) 크기의 이미지를 (128, 128, 3) 형태로 변환
    image_t = image.transpose(2, 0, 1)
    array_transposed1 = np.transpose(image_t, (1, 2, 0))

    # 이미지 회전: 시계 방향으로 90도 회전 (실제로는 반시계 방향으로 세 번 회전)
    image1 = np.rot90(array_transposed1, k=3)
    # plt.imshow(image1)  # 변환된 이미지를 시각적으로 확인하기 위한 코드
    # plt.show()

    # 마스크 전처리: vis_output_model을 마스크로 사용
    array_transposed2 = vis_output_model

    # 마스크 회전: 시계 방향으로 90도 회전 (실제로는 반시계 방향으로 세 번 회전)
    mask_image1 = np.rot90(array_transposed2, k=3)
    # plt.imshow(mask_image1)  # 변환된 마스크 이미지를 시각적으로 확인하기 위한 코드
    # plt.show()

    # 이미지 및 마스크 데이터 타입 변환: 256을 곱한 후 정수형(uint8)으로 변환
    image1 = image1 * 256
    image1 = image1.astype(np.uint8)
    mask_image1 = mask_image1.astype(np.uint8)

    # PIL 이미지 객체로 변환
    image_source_pil = Image.fromarray(image1)
    image_mask_pil = Image.fromarray(mask_image1)

    # 이미지 및 마스크를 시각적으로 확인
    display(*[image_source_pil, image_mask_pil])

    # 전처리된 이미지와 마스크 반환
    return image_source_pil, image_mask_pil


def generate_image(image, mask, prompt, negative_prompt, pipe, seed, device):
    # 인페인팅을 위한 이미지 크기 조정: 512x512 크기로 이미지와 마스크를 리사이즈
    w, h = image.size
    in_image = image.resize((512, 512))
    in_mask = mask.resize((512, 512))

    # 시드 설정: 재현 가능한 이미지 생성을 위한 시드값 설정
    generator = torch.Generator(device).manual_seed(seed)

    # [1] 빈칸을 작성하시오.
    # 이미지 생성: Stable Diffusion 파이프라인을 사용하여 이미지 생성
    result = 

    # [2] 빈칸을 작성하시오.
    # 생성된 결과 이미지 중 첫 번째 이미지를 추출
    result = 

    # 원래 이미지 크기로 리사이즈하여 결과 반환
    return result.resize((w, h))
