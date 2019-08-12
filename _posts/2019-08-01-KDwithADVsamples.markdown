---
layout: post
title:  "Knowledge Distillation with Adversarial Samples Supporting Decision Boundary 정리"
date:   2019-08-01 21:05:00
author: Sangheon Lee
categories: Paper
use_math: true
---

# Knowledge Distillation with Adversarial Samples Supporting Decision Boundary 정리
- 저자 : Byeongho Heo, Minsik Lee, Sangdoo Yun, Jin Young Choi
- 학회 : AAAI 2019
- 날짜 : 2018.05.15 (last revised 2018.12.14)
- 인용 : 4회
- 논문 : [paper](https://arxiv.org/pdf/1805.05532.pdf)

## 1. Introduction
### 1-1. Knowledge Distillation
- 딥러닝의 대가 Hinton 교수님의 2015년 [논문](https://arxiv.org/pdf/1503.02531.pdf).
- 이미 학습된 DNN (Deep Neural Network) 를 이용하여 새로운 DNN을 학습하는 방법.

  ![image](https://user-images.githubusercontent.com/26705935/62862896-3b2ff280-bd42-11e9-9db1-416e5bfe5dc3.png)

  - **이미 학습된 DNN = Teacher network (large)**
  - **새로 학습할 DNN = Student network (small)**

- **학습 방법**

  (1) Teacher network를 학습한다.

  (2) 각 data에 대해, teacher network를 통해 얻는 *classification probability* (softmax 이전 layer의 output) 를 이용하여 *temperature probability* 를 계산 및 저장한다.

    - **Temperature probability**

    ![image](https://user-images.githubusercontent.com/26705935/62863147-ed67ba00-bd42-11e9-8260-7498a0bdab8d.png)

    - 증류 (distillation) 에서 temperature 용어를 따옴.

  (3) Student network는 기존 데이터의 *original labels* 와 (2)에서 구한 *temperature probability*를 이용하여, *KD loss* (*Knowldege Distillation loss*) 를 통해 학습한다.

    - **KD loss**

    ![image](https://user-images.githubusercontent.com/26705935/62863558-176dac00-bd44-11e9-98ff-594a6a2969ba.png)

- **결과**
  - MNIST 데이터 분류
    - Baseline 모델 (3 layers DNN, 784-800-800-10) : 146개의 test error.
    - Teacher network (3 layers DNN, 784-1200-1200-10) : 67개의 test error.
    - Student network with teacher (Baseline과 같은 구조) : 74개의 test error.
    - 쌩으로 학습했을 때 (146개) 보다, teacher를 사용했을 때 (74개) 더 성능이 좋음.

  - MNIST without "3"
    - 위의 teacher를 이용하여, "3"에 해당하는 데이터 없이 student network를 학습.
    - 결과는 109개의 test error. 특히 1010개의 "3" test data 중 14개만 틀림.
    - Classification probability를 통해 학습하기 때문에, 직접적인 label 데이터가 없어도 **label간의 상관관계를 어느 정도 파악할 수 있음.**

  - MNIST with only "7" and "8"
    - "7"과 "8"에 해당하는 데이터만으로 student network를 학습.
    - 무려 13.2%의 test error. 엄청난 성능을 보임.

  - 결과적으로, student network는 teacher network보다 **작지만 비슷한 성능**을 보일 수 있다.

- **사용**
  - 주된 용도: 거의 동등한 성능을 보이면서, **네트워크의 크기를 줄이고자 할 때 사용.**
  - Distillation으로 학습한 student 모델이, adversarial attack (적대적 공격) 에 강인한 모습을 보인다는 [논문](https://arxiv.org/pdf/1511.04508.pdf)이 2016년 발표됨. (*Defensive Distillation*)

- **문제점**
  - Distillation에 관한 기존 연구들은 대부분, 위의 *KD loss*를 상황 및 데이터에 알맞게 변형한 loss들을 제안하였음.
  - 이는 단순한 수식의 변형일 뿐더러, 효과 (성능 상승) 가 미미했고, 특정 상황 혹은 데이터에 specific하다는 한계를 보임.

- 이 논문에서는 더욱 효과적인 knowledge distillation 방법을 제안함.
  - **모델의 decision boundary (결정 경계) 근처에 있는 데이터**를 이용.
  - 특히,  **adversarial attack**은 특정 데이터를 모델의 decision boundary 근처로 이동하게 함.
  - 즉, adversarial attack을 기반으로 **Boundary Supporting Sample (BSS)** 를 생성.

## 2. Related Works
### 2-1. Adversarial Attack
