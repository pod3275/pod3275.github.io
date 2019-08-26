---
layout: post
title:  "Using Pre-Training Can Improve Model Robustness and Uncertainty 정리"
date:   2019-08-01 21:05:00
author: Sangheon Lee
categories: paper
use_math: true
---

# Using Pre-Training Can Improve Model Robustness and Uncertainty 정리
- 저자 : Dan Hendrycks, Kimin Lee, Mantas Mazeika
- 학회 : ICML 2019
- 날짜 : 2019.01.28 (last revised 2019.06.21)
- 인용 : 9회
- 논문 : [paper](https://arxiv.org/pdf/1901.09960.pdf)

## 1. Introduction
### 1-1. Pre-Training
- Pre-training이란 내가 원하는 task 이외의 다른 task의 데이터를 이용하여 주어진 모델을 먼저 학습하는 과정을 말함.
- 특히 이미지 데이터에 대한 task를 수행하는 모델의 경우, ImageNet 데이터를 이용한 pre-training을 널리 사용함.

- Pre-training의 연구 배경
  - ([다음 블로그 내용](https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220884698923&proxyReferer=https%3A%2F%2Fwww.google.com%2F)을 참고함.)
  - 2006년 이전에는 hidden layer가 2개 이상인 neural network (이하 딥러닝) 의 경우 학습이 제대로 이루어지지 않아, 널리 사용하지 못하였음.
  - 딥러닝의 학습을 위해 제안된 것이
