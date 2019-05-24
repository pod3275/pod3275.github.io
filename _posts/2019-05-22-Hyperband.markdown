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
- One-armed bandit, 외팔이 강도
  - 하나의 레버를 가지고 있는 슬롯머신을 일컫는 말.
  -
