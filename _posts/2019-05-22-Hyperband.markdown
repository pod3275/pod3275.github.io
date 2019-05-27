---
layout: post
title:  "Hyperband; A Novel Bandit-Based Approach to Hyperparameter Optimization 정리"
date:   2019-05-22 21:20:00
author: Sangheon Lee
categories: Paper
---

# Hyperband; A Novel Bandit-Based Approach to Hyperparameter Optimization 정리
- 저자 : Lisha Li, Kevin Jamieson, Giulia DeSalvo,Afshin Rostamizadeh, Ameet Talwalkar
- 학회 : JMLR 2018
- 날짜 : 2016.05.21 (last revised 2018.06.18)
- 인용 : 198회
- 논문 : [paper](https://arxiv.org/pdf/1603.06560.pdf)

## 1. Introduction
### 1-1. Hyperarameter
- 모델의 hyperparameter란 학습 과정에 의해 변하지 않는 값으로 모델의 구조, 학습 과정 등을 정의함.
  - ex) # of layers, # of hidden nodes, learning rate, l2 regularization lambda
- 주어진 모델에 대해 최고의 성능을 내도록 하는 hyperparameter는 모델 type, 데이터 종류 등의 환경에 따라 매우 다름.
  - 즉, 무슨 환경에서든 항상 최적인 hyperparameter 값은 존재하지 않음.
- 또한 학습을 끝낸 모델의 성능은 hyperparameter 설정에 따라 천차만별임.

  ![image](https://user-images.githubusercontent.com/26705935/58179450-2fa0d280-7ce3-11e9-8fb1-caf5e08b802c.png)

  - 그림: 모델의 hyperparameter 설정에 따른 성능의 변동

- 따라서 특정 기계 학습을 잘 쓰려면, 주어진 환경에서 최적의 hyperparameter 설정은 필수적임.
- 기존에는 하나하나 찾아보거나 (소위 trial-and-error) 구간을 나누어서 찾아봤지만 ([grid search](https://towardsdatascience.com/grid-search-for-model-tuning-3319b259367e)) 시간이 너무 오래 걸리고, 결과 모델의 성능도 좋진 않음.
- 따라서 모델의 hyperparameter를 최적화하는 기법에 관한 연구가 진행됨.

### 1-2. Hyperparameter optimization
- Bayesian Optimization
  - 가장 유명한 hyperarameter 최적화 기법
  - 모델의 hyperarameter에 따른 모델의 성능 함수를 확률 모델로 regression하고, 모델의 성능이 높을 것으로 기대되는 hyperarameter 설정 point를 도출하여 탐색함. (한글로 잘 정리되어있는 블로그 [참고](http://research.sualab.com/introduction/practice/2019/02/19/bayesian-optimization-overview-1.html))
  - 장점 : (이전 정보를 활용하기 때문에) 결과 모델의 성능이 높다.
  - 단점 : 오래걸린다.
    - 기본적으로 탐색이 순차적으로 진행됨. (탐색하고, 확률 모델 update하고, 다음 탐색 point 찾고, ...)
    - 확률 모델 regression할 때 Gaussian Process Regression을 사용하는데, GP Regression의 time complexity가 관측한 데이터의 세제곱임. (후에 다른 regression 기법을 적용한 [TPE](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)가 제안됨)
- 저자가 말하는 Bayesian Optimization의 단점
  - 일반적으로 모델 학습할 때, accuracy 혹은 loss의 변동을 보면서 성능이 높을 모델이다 아니다를 판단할 수 있음.
  - 그런데 Bayesian Optimization은 특정 budget(epoch, data등 학습에 투입되는 자원, epoch라고 봐도 무방함)만큼을 반드시 소모하여 학습을 일정한 수준까지 해야함
  - 즉, budget의 낭비로 인해 탐색 시간이 길다.
- 새로운 hyperparameter optimization 기법 제시
  - 모델 학습 과정에서 중간 accuracy 혹은 loss를 보고, 좋을 것으로 예상되는 모델을 선출 및 선출된 모델에 더 많은 budget을 할당하자.
  - Hyperparameter optimization problem을 multi-armed bandit problem로 대치.

## 2. Backgrounds
### 2-1. Multi-armed bandit problem
- One-armed bandit (외팔이 강도)

  ![image](https://user-images.githubusercontent.com/26705935/58404464-211a3880-80a0-11e9-8c1c-0d593fb5cf57.png)

  - 하나의 레버를 가지고 있는 슬롯머신을 일컫는 말.
- Multi-armed bandit problem
  - 여러 개의 슬롯 머신(arms)을 당길 수 있는 상황.
  - 각각의 슬롯 머신을 당겨서 얻을 수 있는 reward는 서로 다름.
  - Reward는 어떤 확률 분포에 의해 draw되는 random variable임.
  - 제한 시간 내에 (혹은 제한 횟수 내에) 최대의 reward를 얻기 위해서는 슬롯 머신을 어떤 순서로 당겨야 할까?
- 문제는 arm마다 보상이 다르고, 한 번의 당김에서 하나의 arm의 reward 값만 관측 가능하다는 것.
- Exploration vs exploitation

  ![image](https://user-images.githubusercontent.com/26705935/58408165-38f5ba80-80a8-11e9-95db-efb6bb385e7f.png) (사진 [출처](https://medium.com/user-experience-ux-experts/you-probably-dont-know-how-to-really-create-great-experiences-e991fbc56767))

  - 최적화 문제에서 대두되는 두 가지 중요한 요소.
  - Exploration: 더 높은 reward를 내는 슬롯 머신을 찾기 위해, 기존에 당기지 않은 새로운 슬롯 머신을 당겨보는 것.
  - Exploitation: 높은 reward를 얻기 위해, 지금까지 당긴 슬롯 머신 중 가장 높은 reward를 내는 머신을 다시 당기는 것.
  - Exploration과 exploitation은 서로 trade-off 관계에 있음.
  - 따라서 두 가지 요소를 조화롭게 적용하는 최적화 정책(policy)은 필수적임.

### 2-2. Non-stochastic Best Arm Identification
- 이 논문은 아니고, 같은 저자의 [이전 논문](https://arxiv.org/pdf/1502.07943.pdf) 내용임.
- Best arm identification problem
  - (Multi-armed banit problem) 제한 시간 내에 최대의 reward 얻기.
  --> (Best arm identification problem) 최소의 regret을 내는 arm을 찾기.
- 문제의 환경 분류: Stochastic and Non-stochastic setting
  - Stochastic setting

    ![image](https://user-images.githubusercontent.com/26705935/58405341-2d06fa00-80a2-11e9-9d43-47fc93dfa9f8.png)

    - 각 arm의 regret이 수렴한다.(converge)
    - 수렴하는 정도(convergence rate)를 알고 있다.

  - Non-stochastic setting

    ![image](https://user-images.githubusercontent.com/26705935/58405393-46a84180-80a2-11e9-9039-fd75a150dcbf.png)

    - 각 arm의 regret이 수렴한다.
    - 수렴하는 정도(convergence rate)를 모른다.
    - 하나의 arm을 당기는 cost는 매우 높다.
  - 하이퍼파라미터 최적화 문제는 non-stochastic setting과 유사함.

### 2-3. Multi-armed bandit problem과 하이퍼파라미터 최적화 문제
- Best arm identification problem --> 하이퍼파라미터 최적화
  - Arms = 하이퍼파라미터 설정들
  - number of pulling the arm = number of training iterations of 하이퍼파라미터 설정
  - regret = 중간까지 학습한 모델의 (intermediate) validation loss
- 즉, regret의 최종 수렴 값이 가장 낮은 arm을 찾는다. == 최종 loss가 가장 낮은 하이퍼파라미터 설정을 찾는다.

## 3. Proposed Methods
### 3-1. Successive Halving Algorithm (SHA)
- 본 논문의 제안 기법은 아니고, 저자들의 [이전 논문](https://arxiv.org/pdf/1502.07943.pdf)에서 제안한 하이퍼파라미터 최적화 해결책.
- 제한된 시간에서 최소의 loss를 갖는 모델의 하이퍼파라미터 설정을 찾는 것이 목표.

  ![image](https://user-images.githubusercontent.com/26705935/58406124-e0242300-80a3-11e9-91ab-0033790cb037.png)

  1. 총 탐색에 소요되는 budget 설정. (B)
  2. n개의 하이퍼파라미터 설정을 랜덤하게 뽑음. (Sk)
  3. S0의 모델들에 동일한 budget을 할당. (rk)
  4. 학습 및 중간 loss 추출.
  5. 중간 loss를 기준으로, 성능이 좋지 않은 하이퍼파라미터 설정을 반 만큼 버림. (Sk+1)
  6. 하나의 하이퍼파라미터 설정이 남을 때까지 2, 3, 4, 5를 반복.

- 이해가 안가면 숫자를 대입해보면 됨.

  ![image](https://user-images.githubusercontent.com/26705935/58406834-73118d00-80a5-11e9-86e9-3a9dbf4213ae.png)

  - 랜덤하게 16개를 뽑아서 1 epoch 만큼 학습하고 좋은 8개를 추출함.
  - 추출된 8개를 2 epochs 만큼 학습하고 (1 epoch 만큼 더 학습) 좋은 4개를 추출함.
  - 추출된 4개를 4 epochs 만큼 학습하고 (2 epochs만큼 더 학습) 좋은 2개를 추출함.
  - 추출된 2개를 8 epochs 만큼 학습하고 (4 epochs만큼 더 학습) 좋은 1개를 추출함. --> 결과!

- 이게 왜 수렴하는가?

  ![image](https://user-images.githubusercontent.com/26705935/58407899-ac4afc80-80a7-11e9-9001-545d74d87457.png)

  - 최종 loss(수렴 값)과 현재 loss의 차이에 대한 함수가 non-increasing function이라고 가정.
  - 특정 t 이상의 budget을 할당하여 학습된 모델의 중간 loss를 비교하는 것만으로도, 최종 loss의 대소관계를 알 수 있다는 것을 증명.
  - 그렇다면 대소관계가 반영되는 t는 얼마인지 어떻게 알 수 있는가?
    - 이에 대해 총 소요 budget B를 충분히 크게 잡으면 best arm이 보장된다는 것을 증명함. (생략)
  - 총 소요 budget을 크게 잡기 위해 doubling trick을 적용.
    - 말 그대로 그냥 B를 2B로 하여 돌리고, 3B로 하여 돌리고, ...

- SHA의 단점
  - 알고리즘 자체의 hyperparameter(input) : B와 n.
  - B와 n(정확히는 B/n)에 따라서 exploration과 exploitation의 비율이 정해짐.
  - 따라서 알고리즘 성능을 위해 B와 n이라는 hyperparameter 설정이 굉장히 중요해짐.

- 그래서 이 논문에서 제안한 것이 ***Hyperband*** 입니다. (이제야 이 논문을 처음 언급;; )

### 3-2. Hyperband
