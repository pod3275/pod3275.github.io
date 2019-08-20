---
layout: post
title:  "Knowledge Distillation with Adversarial Samples Supporting Decision Boundary 정리"
date:   2019-08-01 21:05:00
author: Sangheon Lee
categories: paper
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
  - 일반적으로 large network에서 small network로 knowledge transfer가 이루어짐.

- **학습 방법**

  (1) Teacher network를 학습한다.

  (2) 각 data에 대해, teacher network를 통해 얻는 *classification probability* (softmax 이전 layer의 output) 를 이용하여 ***temperature probability*** 를 계산 및 저장한다.

    - **Temperature probability**

    ![image](https://user-images.githubusercontent.com/26705935/62863147-ed67ba00-bd42-11e9-8260-7498a0bdab8d.png)

    - 증류 (Distillation) 에서 temperature 용어를 따옴.

  (3) Student network는 기존 데이터의 *original labels* 와 (2)에서 구한 *temperature probability*를 이용하여, ***KD loss*** (***Knowldege Distillation loss***) 를 통해 학습한다.

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
  - 주된 용도: 거의 동등한 성능을 보이면서, **모델의 크기를 줄이고자 할 때 사용.**
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
  - 일종의 distillation 에서의 data augmentation.

- **Boundary Supporting Sample (BSS)**: teacher network의 decision boundary 근처에 존재하는 데이터.
  - Adversarial example과 비슷한 개념이고, 생성 방식도 비슷하지만, 약간 다름.

- 제안 기법 4가지 구성
  - Iterative scheme to find BSS.
  - Knowledge Distillation using BSS.
  - BSS 기반 KD와 관련한 다양한 issue들.
  - 두 모델의 decision boundary의 유사도를 측정하는 metrics.

### 3-1. Iterative Scheme to Find BSS

  ![image](https://user-images.githubusercontent.com/26705935/62943986-b1058e00-be16-11e9-81a9-bf75b112fdfd.png)

- 그림과 같이, teacher network를 기반으로 특정 데이터 (base sample) 로부터 adversairl sample을 생성함.
- 생성은 white-box attack 방식과 같이, x에 따른 loss의 gradient를 통한 gradient descent method.

- **BSS 생성의 loss function**

  ![image](https://user-images.githubusercontent.com/26705935/62944397-88ca5f00-be17-11e9-9328-1b50a75fcd93.png)

  - b: base sample의 class, k: (b가 아닌) target class.
  - $f_b(x)$: x의 classification score (softmax 이전 값들) 중 b class에 해당하는 값.
  - $f_k(x)$: x의 classification score (softmax 이전 값들) 중 k class에 해당하는 값.

  - Loss를 최소화함 == b class 확률보다 k class 확률을 높임 == 결과가 k class가 되도록 함.

- **GDM with loss**

  ![image](https://user-images.githubusercontent.com/26705935/63001739-3e94bc80-beaf-11e9-975a-1bf848828367.png)

  - (- 항의 $\eta$ 앞쪽 곱): gradient 크기, (가장 왼쪽 분수): gradient의 방향.
  - $\epsilon$: loss가 (-)가 되게 하기 위함 == x가 decision boundary를 넘어가게 하기 위함.

    ![image](https://user-images.githubusercontent.com/26705935/63002088-16598d80-beb0-11e9-8c26-3fc1989811e6.png)

    - 테일러 급수를 이용하여 정리해보면 위와 같이 negative loss가 가능해짐.

- **GDM step을 멈추는 조건**

  ![image](https://user-images.githubusercontent.com/26705935/63004025-705c5200-beb4-11e9-8d39-837d74bcdff9.png)

  - (a): loss가 (-)가 되면. BSS로 채택함. (accept)
  - (b): x가 다른 class 경계 안으로 들어가면. BSS가 아니므로 버림. (reject)
  - (c): 너무 많은 step을 가면. 그 쪽 decision boundary가 너무 멀기 때문에 버림. (reject)

### 3-2. Knowledge Distillation using BSS

- 이미 학습된 teacher classifier ($f_t$) 를 이용한, student classifier ($f_s$) 의 학습 loss $L(n)$

![image](https://user-images.githubusercontent.com/26705935/63003443-17d88500-beb3-11e9-899a-6c67bd6af5f7.png)

  - $L_{cls}(n)$: 원본 데이터 ($(x_n, c_n)$) 의 hard label (true label, $y^{true}$) 을 학습.
  - $L_{KD}(n)$: 원본 데이터에 대한 $f_t$의 temperature probability를 학습.
  - **$L_{BS}(n, k)$: BSS ($\dot{x}_n^k$) 에 대한 $f_t$의 temperature probability를 학습.**
    - Decision boundary 정보를 갖는 BSS를 학습에 이용함.

  - $\alpha, \beta$ 는 loss의 영향력을 나타내는 hyperparameter이고, $p_n^k$는 target class k 가 선택될 확률임. (3-3에서 설명)

### 3-3. Various Issues on using BSSs
- **How to choose Base Sample?**
  - 모든 데이터를 base sample로 하여 각각의 BSS를 구하는 게 아니고, 괜찮을 것으로 보이는 애들만 선택하여 그에 대한 BSS를 구함.

  - 선택되는 base sample $C$ 의 조건:

  ![image](https://user-images.githubusercontent.com/26705935/63004954-8539e500-beb6-11e9-9e22-33313e746688.png)

  - 즉, BSS를 생성하는 기준인 base sample은 teacher network가 맞추고, (학습 과정 속 현재의) student network도 맞추는 데이터.
  - Student network 학습 batch 내에서, 위와 같은 조건을 만족하는 N개의 base sample을 뽑음.

- **How to choose Target Class k?**
  - BSS를 생성하는 과정에서, target class는 아래와 같은 확률 분포 하에서 하나를 sampling 함.

  ![image](https://user-images.githubusercontent.com/26705935/63005358-4ce6d680-beb7-11e9-9e6b-44397f712582.png)

  - $p_n^k$: target class로 k가 선택될 확률.
    - Teacher network를 기준으로 base class가 아닌 다른 class들의 probability 비율.

  - 즉, base class와 비슷하다고 판단되는 (가장 가까운) class를 target class로 잡음.

### 3-4. Metrices for Similarity of Decision Boundaries
- 두 decision boundary의 유사도를 측정하는 두 가지 metrics 제안.

- Magnitude Similarity (***MagSim***) and Amgle Similarity (***AngSim***)

  ![image](https://user-images.githubusercontent.com/26705935/63005847-4a38b100-beb8-11e9-92bf-f0ee75408fcf.png)

  - $\bar{x}_n^{k, t} = \dot{x}_n^{k, t} - x_n$ : Perturbation vector.
  - *MagSim*, *AngSim* $\in [0, 1]$, 높을수록 더 유사함.

  - Base sample로부터 두 모델의 decision boundary까지를 이은 vector들의 크기 및 각도를 비교.

## 4. Experiments
- 이미지 분류 모델 Knowledge Distillation 실험
  - 데이터: CIFAR10, ImageNet (32*32), TinyImageNet

- 비교 기법
  - Original: Classification loss (원래 데이터의 true label) 만으로 학습함.
  - Hinton: Classification loss + (기존) KD loss 로 학습함.
  - FITNET, AT, FSP: Hinton 기법에 부가적인 요소를 추가한 distillation 기법들.
  - FSP: layer-wise correlation matrix를 이용하는 기법으로, 실험에서 본 논문 제안 기법과 결합함.

- Teacher, student network 종류

  ![image](https://user-images.githubusercontent.com/26705935/63018844-c2ad6b00-bed4-11e9-870a-3f16ee017063.png)

- **결과 Student 분류 성능**

  ![image](https://user-images.githubusercontent.com/26705935/63018938-0607d980-bed5-11e9-8dbd-5a2e86555211.png)

  - 기존의 distillation 기법들에 비해 제안 기법 혹은 FSP와 결합한 기법의 성능이 높음.

- **Student의 generalization 평가**
  - 가정: **적은 데이터 만으로 높은 성능을 보인다면 generalization을 잘 한 것이다.**
  - CIFAR10 학습 데이터의 양을 100%에서 20%까지 줄이면서 student의 성능 평가.

  ![image](https://user-images.githubusercontent.com/26705935/63019191-98a87880-bed5-11e9-941c-3fd7892a7c4c.png)

  - 그래프: 학습 데이터 양에 따라, Original 기법과 비교한 성능 개선 정도.

  - 학습 데이터가 적은 상황에서, student를 그냥 학습한 것에 비한 성능 개선율이 매우 높음.

- **두 decision boundary 간의 유사도 측정**
  - *MagSim* and *AngSim* using CIFAR10 dataset.

  ![image](https://user-images.githubusercontent.com/26705935/63020465-28035b00-bed9-11e9-8e0e-3f6f477a4048.png)

  - Original 기법이나 Hinton 기법에 비해 높은 유사도를 보임.
    - *다른 기법들은 왜 비교하지 않았는가?*

- **BSS == Adversarial attack?**
  - BSS를 찾는 과정을, 기존의 adversarial attack 기법으로 대체하면 어떻게 되는가?
  - 즉, BSS = adversarial example 이며, distillation에 adversarial training을 적용한 것.

  ![image](https://user-images.githubusercontent.com/26705935/63019599-ccd06900-bed6-11e9-8ed2-59a12a9b513f.png)

  - 실험 결과 제안 기법이 기존 adversarial attack을 적용한 것보다 우수한 distillation 성능을 보임.

## 5. Conclusion
- *Knowledge Distillation* 이란 이미 학습된 모델을 이용하여 새로운 모델을 효율적으로 학습하는 기법임.
  - 일반적으로 이미 학습된 모델 (*teacher network*) 은 크기가 크고, 새로 학습할 모델 (*student network*) 은 크기가 작은 것으로 설정함.
  - 비슷한 성능을 보이면서 **모델의 크기를 줄일 수 있음.**

- 본 논문에서는 모델의 decision boundary 정보를 포함하고 있는 데이터를 생성 및 이용하는, 더욱 효과적인 distillation 기법을 제안함.
  - 적대적 공격 (Adversarial attack) 개념을 기반으로 ***Boundary Supporting Sample (BSS)*** 를 생성.
  - 생성된 BSS를 이용하여 student network를 학습하는 distillation loss 제안.
  - 두 모델의 decision boundary의 유사도를 측정하는 두 measures 제안.

- Knowledge Distillation 결과 모델의 정확도 및 일반화 성능을 높임.
