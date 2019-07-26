---
layout: post
title:  "Fast AutoAugment 정리"
date:   2019-07-12 21:50:00
author: Sangheon Lee
categories: Paper
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
- [TPE](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) (Bayesian Optimization과 비슷한 black-box optimization 기법) 기반의 빠르고 효과적인 augmentation policy search 기법.

### 2-1. Search Space
- Operation *O*
  - Augmentation 기법 단위.
  - 각 operation은 확률 *p* 와 세기 $\lambda$ 값을 가짐. (*p*, $\lambda$ $\in$ [0,1])

- Sub-policy $\tau$ $\in$ S
  - $N_\tau$ 개의 operation들.

  ![image](https://user-images.githubusercontent.com/26705935/61947495-449a2a80-afe0-11e9-9dc0-a737071e794a.png)

  - 이미지에 적용 시, 각 operation을 확률에 따라 순서대로 적용.
  - 하나의 sub-policy = 하나의 이미지 생성. (위의 그림에서, 오른쪽 4장의 이미지 중 하나.)

- Policy $\Tau$
  - $N_\Tau$ 개의 sub-policy들.
  - 하나의 policy = $N_\Tau$ 개의 이미지 생성.
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
  - $\Tau_* = \argmax_T{R(\theta^*|\Tau(D_{A}))}$
    - $\theta^*$ : $D_M$ 으로 학습한 모델의 parameter.
    - $R(\theta|D)$ : 데이터 D의 모델 $\theta$ 에 대한 정확도(accuracy).
  - 즉, $D_M$ 으로 학습한 모델을 기준으로, augmented $D_{A}$ 에 대한 성능이 높은, 그런 policy를 찾자.

- 기존 Augmentation 개념과 반대로 생각함.
  - 기존 개념: **학습 데이터에 augmentation을 적용**한 데이터로 학습된 모델을 기준으로, 검증 데이터에 대한 성능이 높은 augmentation policy가 최적.
  - 제안 개념: 학습 데이터로 학습된 모델을 기준으로, **검증 데이터에 augmentation을 적용**한 데이터에 대한 성능이 높은 augmentation policy가 최적.

  - **이렇게 하면, 모델을 재학습할 필요가 없음 : 시간 단축 가능.**

### 2-3. Algorithm

  ![image](https://user-images.githubusercontent.com/26705935/61948934-2cc4a580-afe4-11e9-9d8d-f3b311189bfc.png)

  ![image](https://user-images.githubusercontent.com/26705935/61948960-3f3edf00-afe4-11e9-9948-4ab5472ba92b.png)

- 3단계
