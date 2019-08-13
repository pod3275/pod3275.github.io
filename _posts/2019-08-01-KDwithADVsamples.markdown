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

    - 증류 (Distillation) 에서 temperature 용어를 따옴.

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
  - Knowledge Distillation으로 학습한 student 모델이, adversarial attack (적대적 공격) 에 강인한 모습을 보인다는 [논문](https://arxiv.org/pdf/1511.04508.pdf)이 2016년 발표됨. (*Defensive Distillation*)

- **문제점**
  - Knowldege Distillation에 관한 기존 연구들은 대부분, 위의 *KD loss*를 상황 및 데이터에 알맞게 변형한 loss들을 제안하였음.
  - 이는 단순한 수식의 변형일 뿐더러, 효과 (성능 상승) 가 미미했고, 특정 상황 혹은 데이터에 specific하다는 한계를 보임.

- 이 논문에서는 더욱 효과적인 Knowledge Distillation 방법을 제안함.
  - **모델의 결정 경계 (decision boundary) 근처에 있는 데이터**를 이용.
  - 특히,  **adversarial attack**은 특정 데이터를 모델의 decision boundary 근처로 이동하게 함.
  - 즉, adversarial attack을 기반으로 **Boundary Supporting Sample (BSS)** 를 생성.

## 2. Related Works: Adversarial Attack

![image](https://user-images.githubusercontent.com/26705935/62939101-6d595700-be0b-11e9-93b8-fe562b6f3d4d.png)

- 2014년 Ian Goodfellow의 [논문](https://arxiv.org/pdf/1412.6572.pdf)에서 처음 언급됨.
- **Adversarial attack (적대적 공격)** : 입력 이미지에 사람이 구분하기 힘든 noise를 섞음으로써 모델로 하여금 결과를 다르게 하는 것.
  - 또는 그러한 이미지를 생성하는 방법.
  - 위의 그림: "panda" label을 갖는 그림에 noise를 추가하여 생성된 그림은 "gibbon" label을 가짐. ([출처](https://arxiv.org/pdf/1412.6572.pdf))

  - 생성된 이미지 = adversarial image (example) = natural image + **noise (perturbation)**

- 분류
  - 공격자의 상황에 따라 두 가지로 나뉨.

  ![image](https://user-images.githubusercontent.com/26705935/62939044-4569f380-be0b-11e9-920f-b1590c528239.png)

  그림: [출처](https://arxiv.org/pdf/1708.03999.pdf)

  **1. White-box attack**

    - 공격자가 모델의 구조, parameter를 알고, 모델의 loss를 통해 gradient를 계산할 수 있는 경우.
    - 일반적으로, **입력에 따른 모델의 loss의 gradient를 계산하여, 이의 반대 방향으로 입력을 update 하는 방식.**

    - ex) *FGSM (Fast Gradient Sign Method)* attack

    ![image](https://user-images.githubusercontent.com/26705935/62938398-f53e6180-be09-11e9-9259-06a3d4b8e027.png)

    - $x^* $ = adversarial image, $x$ = natural image, $J(x, y)$ = cross-entropy loss, $\epsilon$ = step size.
    - FGSM 이후로 다양한 종류의 attack 기법이 제안됨 ([BIM](https://arxiv.org/pdf/1607.02533.pdf), [JSMA](https://arxiv.org/pdf/1511.07528.pdf), [DeepFool](https://arxiv.org/pdf/1511.04599.pdf), [C&W](https://arxiv.org/pdf/1608.04644.pdf), ...)
    - 직관적인 loss, 안정적인 gradient descent method를 기반으로 하기 때문에, 매우 높은 공격 성공률을 보임.

  **2. Black-box attack**

    - 공격자가 모델에 대한 모든 정보를 모르고, 오로지 **특정 입력에 대한 결과만 얻을 수 있는 경우. (query)**

    - 대표적으로 2가지 방식
      - **대체 모델 (substitute model) 을 통한 공격** : 원래 모델로부터 query를 날려 얻은 결과로 새로운 구조의 모델 학습 및 공격 ([논문](https://arxiv.org/pdf/1602.02697.pdf))
      - **Gradient estimation** 을 통한 공격 : 경사의 기울기 구하는 개념으로 gradient를 estimation하여 구함으로써, white-box attack과 같은 방식으로 공격 ([논문](https://arxiv.org/pdf/1805.11770.pdf), 정리)

    - 정확한 수식을 통해 최적화하는 것이 아니기 때문에, 낮은 공격 성공률을 보임.
    - 하지만 대체 모델을 통한 공격은 대부분의 모델에서 통하기 때문에, 방어하기 어려움.

- Discussion
  - Adversarial example이 왜 발생하는가 에 대한 많은 분석들이 나옴.
  - 가장 그럴싸한 분석

    ![image](https://user-images.githubusercontent.com/26705935/62940728-279e8d80-be0f-11e9-8ce3-1ba257ffa9ee.png)

    - Adversarial attack에 강인한 모델 학습 관련한 2017년 [논문](https://arxiv.org/pdf/1706.06083.pdf)에서 설명한 그림.
    - 점: 실제 데이터, 사각형: 사람이 볼 때, 실제 데이터(점)와 구분할 수 없는 영역. (**$l_\infty$ ball**)
    - 모든 데이터는 $l_\infty$ ball 이 존재하기 때문에, 2번째 그림의 아래 별과 같이 실제로는 파란색인데 초록색 class를 나타내는 데이터가 존재함.
    - 이것이 adversarial example임.

  - 즉, **adverarial example은 모델의 결정 경계 (decision boundary) 근처에 존재함.**

    ![image](https://user-images.githubusercontent.com/26705935/62941417-ead39600-be10-11e9-9d05-54735dfa9de2.png)

    - 실제로 white-box attack 과정을 봐도, 매 step 마다 조금씩 움직이다가 모델의 결과가 바뀌는 순간 멈춤.
    - 다르게 생각하면, **adversarial example은 모델의 decision boundary에 대한 정보를 가지고 있음.**
    - 이러한 정보를 활용하여 Knowledge Distillation을 하면 더 잘 할 것.

## 3. Proposed Methods
### Boundary Supporting Sample (BSS)

  ![image](https://user-images.githubusercontent.com/26705935/62943434-6cc5be00-be15-11e9-9d89-e841e68cfb7e.png)

- Student network가 좋은 performance를 갖기 위해선, decision boundary가 teacher network의 decision boundary와 비슷해야 함.
- 이를 위해, 모델의 decision boundary 정보를 가지고 있는 데이터를 사용한 Knowldege Distillation 제안.

- **Boundary Supporting Sample (BSS)**: teacher network의 decision boundary 근처에 존재하는 데이터.
  - Adversarial example과 비슷한 개념이고, 생성 방식도 비슷하지만, 약간 다름.

- 제안 기법 4가지 구성
  - Iterative scheme to find BSS.
  - Knowledge Distillation using BSS.
  - BSS 기반 KD와 관련한 다양한 issue들.
  - 두 모델의 decision boundary의 유사도를 측정하는 metric.

### 3-1. Iterative Scheme to Find BSS

  ![image](https://user-images.githubusercontent.com/26705935/62943986-b1058e00-be16-11e9-81a9-bf75b112fdfd.png)

- 그림과 같이, teacher network를 기반으로 특정 데이터 (base sample) 로부터 adversairl sample을 생성함.
- 생성은 white-box attack 방식과 같이, x에 따른 loss의 gradient를 통한 gradient descent method.

- BSS 생성의 loss function

  ![image](https://user-images.githubusercontent.com/26705935/62944397-88ca5f00-be17-11e9-9328-1b50a75fcd93.png)

  - b: base sample의 class, k: (b가 아닌) target class.
  -
