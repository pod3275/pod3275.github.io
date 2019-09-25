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
- Pre-training이란 내가 원하는 task 이외의 **다른 task의 데이터를 이용하여 주어진 모델을 먼저 학습하는 과정**을 말함.
- 특히 이미지 데이터에 대한 task를 수행하는 모델의 경우, **ImageNet 데이터**를 이용한 pre-training을 널리 사용함.


- 과거의 Pre-training
  - ([다음 블로그 내용](https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220884698923&proxyReferer=https%3A%2F%2Fwww.google.com%2F)을 참고함.)
  - 2006년 이전에는 hidden layer가 2개 이상인 neural network (이하 딥러닝) 의 경우 학습이 제대로 이루어지지 않아, 널리 사용하지 못하였음.
  - 딥러닝의 학습을 위해 제안된 여러 기법들 중 Bengio 교수의 **Greedy Layer-Wise Training**.

  ![image](https://user-images.githubusercontent.com/26705935/65571500-e4077980-df9f-11e9-8aaf-c4b66884571f.png)

  - 딥러닝 모델의 각 layer의 node수와 같은 hidden layer를 하나 갖는 auto-encoder 구조의 분할 모델을 unsupervised way로 학습.
  - 학습된 여러 auto-encoder의 weight (박스 안의 부분) 들을 원래 모델에 합쳐서, 원래 모델을 전체적으로 supervised way로 학습.

  - 이후 dropout, relu 등 딥러닝 학습을 가능하게 하는 다양한 기법들이 제안되면서 현재는 크게 사용하지 않는 기법이지만, 딥러닝 사용을 boosting하는 획기적인 기법이었음.
  - **또한, Pre-train의 개념이 처음으로 제안된 아이디어임.**


- 현재의 Pre-training

  ![image](https://user-images.githubusercontent.com/26705935/65571862-22516880-dfa1-11e9-8ec8-d7a2d7ca040f.png)
  그림: [출처](https://www.mdpi.com/2072-4292/9/7/666/pdf-vor)

  - 주어진 분류 모델을, 내 데이터가 아닌 다른 데이터로 학습.
  - 모델 뒤의 fully-connected layer (dense layer, 분류 또는 개체 감지 등의 결과를 계산하는 부분) 를 내 데이터에 알맞게 바꿔서, 전체 모델을 내 데이터로 학습.

  - 데이터의 feature를 추출하는 것은 task independent하기 때문에, **데이터의 수가 많은 다른 task를 이용하여 모델 앞 단의 feature extractor 부분을 학습하자** 라는 것임.


- 이렇게 널리 쓰이는 Pre-training 기법에 대해, 회의적인 의견을 제시한 논문 등장.
  - "Rethinking ImageNet Pre-training", He et al., 2018. [paper](https://arxiv.org/pdf/1811.08883.pdf) (무려 ResNet 논문의 저자 Kaiming He의 논문)

  - ImageNet pre-trained weight를 사용하는 건 모델 수렴을 빠르게 해준다.
  - 하지만 최종 성능을 높여주거나 overfitting을 방지해주는 역할은 없다.

  ![image](https://user-images.githubusercontent.com/26705935/65572222-5ed19400-dfa2-11e9-997a-997c50d30e7b.png)

  - 그래프와 같이, weight를 randomly initialized한 모델은 학습 속도는 느림.
  - 하지만 오래 학습하다보면 pre-trained weight를 사용한 모델과 성능이 비슷해짐.
  - 특히 오른쪽 그래프처럼, 학습 데이터 수가 적어도 성능이 비슷해지는 걸 보면, generalization 측면에서도 특효가 있지 않음.

  - 최종적으로, pre-trained weight를 사용하는 것이 필수적인 건 아니다. (not necessary)

- 본 논문에서는 위의 논문을 반박함.
  - Pre-trained weight가 성능 개선이 없을 수 있지만, 다른 방면에서 우수성이 있다.
  - 특히, **모델의 robustness 및 uncertainty** 를 크게 개선한다.
    - Robustness: Adversarial Robustness, Label Corruption, Class Imbalance
    - Uncertainty: Out-of-Distribution Detection, Calibration

  - 이에 대한 다양한 실험을 진행하며, pre-trained weight의 우수성을 입증함.
