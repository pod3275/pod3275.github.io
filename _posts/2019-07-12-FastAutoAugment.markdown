---
layout: post
title:  "Fast AutoAugment 정리"
date:   2019-07-12 21:50:00
author: Sangheon Lee
categories: paper
use_math: true
---

# Fast AutoAugment 정리
- 저자 : Sungbin Lim, Ildoo Kim, Taesup Kim, Chiheon Kim, Sungwoong Kim
- 학회 : ICML 2019 (AutoML Workshop)
- 날짜 : 2019.05.01 (last revised 2019.05.25)
- 인용 : 2회
- 논문 : [paper](https://arxiv.org/pdf/1905.00397.pdf)

## 1. Introduction
### 1-1. Data Augmentation
- **Augmenation = Generalization = Avoid Overfitting**
  - Overfitting = 모델이 학습 데이터를 너무 따라가서, test 성능이 낮게 나타나는 경우. (조금 더 자세한 설명은 [여기](https://pod3275.github.io/paper/2019/05/30/Dropout.html))
  - 일반적으로 학습 데이터의 개수가 많으면, overfitting을 피하기 쉽다. = 데이터 manifold를 일반화하기 쉽다.

  ![image](https://user-images.githubusercontent.com/26705935/61942756-06e3d480-afd5-11e9-92b2-53867c7b72ca.png)

  - 그림 [출처](https://hackernoon.com/memorizing-is-not-learning-6-tricks-to-prevent-overfitting-in-machine-learning-820b091dc42)

  - 따라서, 학습 데이터의 양과 다양성을 늘려서 generalization을 이룩하자 = **Augmenatation**

  ![image](https://user-images.githubusercontent.com/26705935/61942967-67731180-afd5-11e9-8e07-9ab7cd0705a3.png)

  - 그림 [출처](https://www.kakaobrain.com/blog/64)
  - 그림과 같이, 한 장의 고양이 사진을 이용하여 다양한 고양이 사진들을 생성할 수 있음.
  - 다양한 augmentation 기법들([Cutout](https://arxiv.org/pdf/1708.04552.pdf), [GAN 기반 기법](https://arxiv.org/pdf/1803.01229.pdf) 등)이 제안되었고, 모델의 성능을 높이고 있음.

- 그렇다고 너무 막 생성하면 안됨.
  - 모델 성능을 최대로 높이려면 augmentation도 잘 해야한다.
  - 즉, 전문가의 지식이 필요하다. (여기서, hyperparameter tuning과 비슷한 모습을 보임)

### 1-2. AutoAugment
- 2018년 Google Brain [논문](https://arxiv.org/pdf/1805.09501.pdf)

![image](https://user-images.githubusercontent.com/26705935/61943777-27ad2980-afd7-11e9-8a16-d6d4a7ac192a.png)

- RNN (Recurrent Neural Network) + RL (Reinforcement Learning)

  (1) Augmentation 기법을 출력하는 RNN controller 생성.

  (2) 이를 통해 얻은 augmentation 기법을 학습 데이터에 적용.

  (3) 모델을 학습 및 성능을 평가하여 reward(R)를 얻음.

  (4) 계산된 reward를 통해 RNN controller 학습.

- Augmentation 기법을 policy, sub-policy 단위로 나누어 search space를 체계화함.

  ![image](https://user-images.githubusercontent.com/26705935/61944104-d9e4f100-afd7-11e9-815f-ef55113a9ac4.png)

  - Fast AutoAugment에도 비슷한 단위로 적용됨.

- 약간 NAS (Neural Architecture Search), 특히 [ENAS](https://arxiv.org/pdf/1802.03268.pdf)와 방식이 비슷함.
- 성능 개선은 매우 높지만 (몇몇은 SOTA 갱신), **시간이 너무 오래걸린다** 는 단점이 있음.

  ![image](https://user-images.githubusercontent.com/26705935/61944699-457b8e00-afd9-11e9-85d1-bbf46f956048.png)

  - RNN 한 번 업데이트를 위해 분류 모델을 full로 학습시켜야 함.
  - 몇 천 GPU 시간이 걸림.

### 1-3. PBA (Population Based Augmentation)
- 2019년 arXiv [논문](https://arxiv.org/pdf/1905.05393.pdf)
- 기존의 Hyperparameter optimization 기법 중, [PBT](https://pod3275.github.io/paper/2019/03/19/PBT.html)(Population Based Training) 알고리즘을 기반으로 함.

  ![image](https://user-images.githubusercontent.com/26705935/51183079-8d777500-1913-11e9-958e-b26d1f285c6f.png)

  (1) { 동일한 모델 + 다른 augmentation 기법 적용 } X 여러 개 를 동시에 학습.

  (2) 중간 지점에서 각 모델의 성능을 비교.

  (3) 성능이 높은 모델의 parameter를 복제하고 (exploit), 적용된 augmentation 기법에 약간의 변형을 줌. (explore)

  (4) 동시 학습 진행. (2)와 (3)을 반복.

- 결과

  ![image](https://user-images.githubusercontent.com/26705935/61946652-23383f00-afde-11e9-8798-d4b9f018f7eb.png)

  - AutoAugment 보다 높은 성능 개선 및 짧은 실행 시간 기록.

## 2. Proposed Method
### Fast AutoAugment
- [TPE](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) (Bayesian Optimization과 비슷한 black-box optimization 기법) 기반의 빠르고 효과적인 augmentation policy search 기법 제안.

### 2-1. Search Space
- Operation *O*
  - Augmentation 기법 단위.
  - 각 operation은 확률 *p* 와 세기 $\lambda$ 값을 가짐. (*p*, $\lambda$ $\in$ [0,1])

- Sub-policy $\tau$ $\in$ S
  - $N_\tau$ 개의 operation들.

  ![image](https://user-images.githubusercontent.com/26705935/61947495-449a2a80-afe0-11e9-9dc0-a737071e794a.png)

  - 이미지에 적용 시, 각 operation을 확률에 따라 순서대로 적용.
  - 하나의 sub-policy = 하나의 이미지 생성. (위의 그림에서, 오른쪽 4장의 이미지 중 하나.)

- Policy $T$
  - $N_T$ 개의 sub-policy들.
  - 하나의 policy = $N_T$ 개의 이미지 생성.
  - 우리가 찾고 싶은 최종.

### 2-2. Search Strategy
- 핵심 개념
  - **Augmenation은 학습 데이터 분포 중 빵꾸난 데이터를 만드는 것.**
  - 즉, train data ($D_{train}$) 와 validation data ($D_{valid}$) 의 데이터 분포(density)를 맞춰주는 역할.
    - $D_{train}$ 에 augmentation 적용 == $D_{valid}$
    - **(반대로 생각해서) $D_{valid}$ 에 augmentation 적용 == $D_{train}$**

- 실제로는 $D_{train}$만 이용해서 augmentation policy 찾을 거니까
  - $D_{train} = D_M \cup D_A$ 로 나눔.
  - 목표: $D_M$ 의 density == Augmented $D_A$ 의 density.

- **데이터의 density 비교**를 어떻게 하는가?
  - **학습된 model 을 이용하자.**
  - $T_* = \arg\max_{T}{R(\theta^{*} \vert T(D_{A}))}$
    - $\theta^{*}$ : $D_M$ 으로 학습한 모델의 parameter.
    - $R(\theta \vert D)$ : 데이터 D의 모델 $\theta$ 에 대한 정확도(accuracy).

  - 즉, $D_M$ 으로 학습한 모델을 기준으로, augmented $D_{A}$ 에 대한 성능이 높은, 그런 policy를 찾자.

- 기존 Augmentation 개념과 반대로 생각함.
  - 기존 개념: **학습 데이터에 augmentation을 적용**한 데이터로 학습된 모델을 기준으로, 검증 데이터에 대한 성능이 높은 augmentation policy가 최적.
  - 제안 개념: 학습 데이터로 학습된 모델을 기준으로, **검증 데이터에 augmentation을 적용**한 데이터에 대한 성능이 높은 augmentation policy가 최적.

  - **이렇게 하면, 모델을 재학습할 필요가 없음 : 시간 단축 가능.**

### 2-3. Algorithm

  ![image](https://user-images.githubusercontent.com/26705935/61948934-2cc4a580-afe4-11e9-9d8d-f3b311189bfc.png)

- 단계

  (1) 학습 데이터 $D_{train}$을 k개의 묶음으로 (class 비율을 맞추어) 나눔. 각각의 묶음은 $D_M$과 $D_A$로 이루어짐.

  (2) $D_M$으로 모델 학습($\theta$) 및 *Bayesian Optimization* 을 통해 $L(\theta \vert T(D_{A}))$ 가 최소가 되는 policy $T$를 search함.

    - $L(\theta \vert T(D_{A}))$ : 모델 $\theta$에 대한 $T(D_{A})$ 데이터의 검증 loss.
    - Bayesian Optimization : [TPE](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) 사용.

  (3) 성능이 좋은 *N*개의 policy들을 병합함. (**$T_*^{(k)}$**)

  (4) (2)와 (3)을 *T*번 반복하여 모든 결과 policy를 병합함.

  (4) 각 k-fold에 대해 (2)~(4)를 반복하여, 모든 결과 policy를 하나로 병합함. (**$T_*$**)

  (5) (4)의 결과를 $D_{train}$에 적용한 augmented data로 모델을 재학습함.

- 알고리즘

  ![image](https://user-images.githubusercontent.com/26705935/61948960-3f3edf00-afe4-11e9-9948-4ab5472ba92b.png)

- 이점
  - **학습된 모델 1개만을 이용**하여 최적의 policy 탐색.
  - 즉, Bayesian Optimization 과정에서, 성능이 높을 것으로 기대되는 augmentation policy를 **뽑아낼 때마다 모델을 학습시킬 필요가 없음.**
  - **탐색 시간이 매우 단축**됨.
  - 또한 search space를 numerical한 공간으로 표현하였기 때문에 (*p*, $\lambda$ $\in$ [0,1]), Bayesian Optimization의 특성과 잘 맞음.

## 3. Experiments
- 4가지 이미지 데이터에 대한 분류 모델에 augmentation 적용.
  - CIFAR-10, CIFAR-100, (reduced) SVHN, (reduced) ImageNet

### 3-1. Hyperparameters 설정
  - Operation 종류 = 16 (Shear X, Rotate, Invert, ...)
  - $N_{\tau}$ (sub-policy 내의 operation 수) = 2
  - $N_{T}$ (policy내 sub-policy 수) = 5
  - k (fold 수) = 5, *T* (각 fold data마다 반복 횟수) = 2
  - B (TPE를 뽑아내는 후보 개수) = 200, *N* (각 반복마다 성능이 좋은 policy 저장할 개수) = 10

  - 즉, 최종적으로 **100개의 policy**를 찾으며, 이에 따라 1장의 data로부터 500장의 augmented data가 생성됨.

### 3-2. 실험 결과
- **정확도 향상**

  ![image](https://user-images.githubusercontent.com/26705935/62288917-63913480-b498-11e9-8dc8-b956517a7590.png)

  - Baseline : Augmentation을 적용하지 않은 것, [Cutout](https://arxiv.org/pdf/1708.04552.pdf) : 가장 널리 사용되는 augmentation 기법
  - [AA](https://arxiv.org/pdf/1805.09501.pdf) : AutoAugment, [PBA](https://arxiv.org/pdf/1905.05393.pdf) : Population Based Augmentation
  - Fast AA의 transfer : Wide-ResNet-40-2 모델과 조금 축소한 데이터를 이용하여 찾은 augmentation 기법들을 그대로 적용한 것.

  - 제안된 기법인 **Fast AA는 Baseline 및 기존 augmentation 기법보다 좋은 성능**을 보임.
  - 또한 **AA 및 PBA보다 높진 않지만, 이에 준하는 성능**을 보임.

- **속도**

  ![image](https://user-images.githubusercontent.com/26705935/62290114-83762780-b49b-11e9-91a2-1fa3c7fe2aa7.png)

  - 이 논문의 핵심 = AutoAugment에 비하여 **탐색 속도의 엄청난 개선**.
  - AA보다 빠르다는 **PBA에 준하는 속도**를 보임. 다음은 PBA 논문에 있는 탐색 속도.

    ![image](https://user-images.githubusercontent.com/26705935/62289185-0649b300-b499-11e9-8c21-02811ccd79eb.png)

  - PBA와 Fast AA의 속도 비교는 (reduced) ImageNet 이용한 실험에서 제대로 비교해봐야 할 것 같음.

## 4. Conclusions
- 딥러닝 모델의 overfitting을 피하기 위한 generalization 기법들 중, 데이터 단계에서 적용할 수 있는 augmentation의 자율 최적화에 관한 연구.
- 기존의 AutoAugment라는 augmentation 최적화 기법은 강화학습을 통해 RNN controller를 학습 구조로서, 탐색 시간이 매우 오래걸린다는 단점이 있음.
- **"Augmentation은 데이터 분포의 빈 공간을 채우는 것"** 이라는 개념 하에, augmetation 기법을 검증 데이터에 적용 및 한 번 학습된 모델로 augmentation 기법 성능 평가.
- 탐색 결과 마다 모델을 학습할 필요가 없기 때문에, 최적화에 소요되는 **총 소요 시간이 감소**함.
- 다양한 이미지 분류 데이터에 대한 실험 결과, AutoAugment 및 PBA에 준하는 성능과 함께 단축된 소요 시간을 보임.

- Auto Augmentation 연구는 후에 **NAS (Neural Architecture Search, 신경망 구조 탐색) 분야에 접목**되어, 모델의 일반화 및 자율 최적화 기법에 관한 연구가 진행될 필요가 있음.

- (개인적인 생각)
  - BO를 뽑아낼 때마다 매 번 학습을 할 필요가 없는 것은 매우 큰 장점인듯 함.
  - 하지만 검증 데이터에 augmentation 기법을 적용하고, 이미 학습된 모델로 loss를 계산하는 것이 과연 그 augmentation 기법에 대한 성능을 100% 반영하는지에 대한 의문이 듦.
  - 두 가지 데이터의 density matching 관점에서 봤을 때 어느 정도 이해는 되지만, 필요충분조건에 대한 수학적인 증명이 필요하다고 생각됨.
