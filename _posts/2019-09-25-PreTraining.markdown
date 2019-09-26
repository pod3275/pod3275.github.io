---
layout: post
title:  "Using Pre-Training Can Improve Model Robustness and Uncertainty 정리"
date:   2019-09-25 20:00:00
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
### Pre-Training
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
  - 하지만 **최종 성능을 높여주거나 overfitting을 방지해주는 효과는 없다.**

  ![image](https://user-images.githubusercontent.com/26705935/65572222-5ed19400-dfa2-11e9-997a-997c50d30e7b.png)

  - 그래프와 같이, weight를 randomly initialized한 모델은 학습 속도는 느림.
  - 하지만 오래 학습하다보면 pre-trained weight를 사용한 모델과 성능이 비슷해짐.
  - 특히 오른쪽 그래프처럼, 학습 데이터 수가 적어도 성능이 비슷해지는 걸 보면, generalization 측면에서도 특효가 있지 않음.

  - **최종적으로, pre-trained weight를 사용하는 것이 필수적인 건 아니다. (not necessary)**


- 본 논문에서는 위의 논문을 반박함.
  - Pre-trained weight가 성능 개선이 없을 수 있지만, 다른 방면에서 우수성이 있다.
  - 특히, **모델의 Robustness 및 Uncertainty** 를 크게 개선한다.
    - Robustness : Adversarial Robustness, Label Corruption, Class Imbalance
    - Uncertainty : Out-of-Distribution Detection, Calibration

  - 이에 대한 다양한 실험을 진행하며, pre-trained weight의 우수성을 입증함.


## 2. Experiments
- 이미지 분류 모델을 이용하여 실험 진행.
- ImageNet 데이터를 이용하여 pre-training한 weight을 사용한 모델과 안한 모델을 비교평가.
- 모델의 robustness 및 uncertainty를 평가함.
  - Robustness: Adversarial Robustness, Label Corruption, Class Imbalance
  - Uncertainty: Out-of-Distribution Detection, Calibration

### 2-1. Adversarial Robustness
- Adversarial attack에 대해 강인한가
  - Adversarial attack 설명: [관련 글](https://pod3275.github.io/paper/2019/08/02/KDwithADVsamples.html) 참고. (Related works 부분)
  - 주어진 adversarial example에 대해, 원래의 class라고 말할 확률 (accuracy) 측정.

- **Adversarial Pre-training** 진행
  - Adversarial training: 모델의 학습 데이터에 (원래 label을 갖는) adversarial example을 추가하는 것.
  - Adversarial pre-training: 원래 모델 학습과정 뿐만 아니라 pre-training 과정에서도
adversarial training을 진행함.

- 실험 결과

  <br>![image](https://user-images.githubusercontent.com/26705935/65689180-6e86d080-e0a7-11e9-9147-a8d0e996439c.png)<br>

  - Adversarial pre-training 기법: clean image에 대한 성능은 약간 떨어졌으나, adversarial example에 대한 정확도가 많이 상승함.
  - ImageNet 데이터셋에서 CIFAR-10과 관련된 class를 지우고 똑같이 진행하더라도 정확도 기준 1.04%p 정도만 떨어짐.
  - CIFAR 데이터셋을 이용한 fine-tuning 과정에서, 모델의 맨 뒷단인 fully-connected layer만 업데이트했을 경우
    - Adversarial example에 대한 정확도: 46.1% (CIFAR-10), 26.1% (CIFAR-100)
    - "*ImageNet pre-training을 통해 adversarial feature를 제대로 학습하여 CIFAR 데이터 도메인으로 transfer하였다.*"

***"Pre-trained weight을 사용한 모델은 Adversarial Attack에 대해 더욱 강인하다."***

### 2-2. Label Corruption
- 모델의 **학습 데이터에 label noise**가 존재
  - Adversarial example도 일종의 noise라고 볼 수 있지만, 여기서는 아예 다른 모양의 이미지지만 label만 이상한 경우를 말함.

  - 입력 $x$, clean label $y$, corrupted label $\tilde y$ 이고, $(x, \tilde y)$가 주어졌을 때,
    - 목표: $argmax_y p(y|x)$

- Deal with Label Corruption problem
  - **Corruption probability matrix $C$ 를 계측** 하는 방향임.
    - $C_{ij} = p(\tilde y = j | y = i)$ : $i$라는 class가 $j$라는 class로 corrupt될 확률.
  - Forward Correction (*Patrini et al., 2017*), GLC (*Hendrycks et al., 2018*) 등

- 실험 결과

  <br>![image](https://user-images.githubusercontent.com/26705935/65689167-62027800-e0a7-11e9-9252-091fd8d874ab.png)<br>

  - Label corruption 기법 + w/ vs. w/o pre-trained weight 의 성능을 비교.
  - 표는 모델의 Test Error Rate임.
  - Pre-trained weight을 사용하였을 때, corrupted data를 학습한 모델의 error rate이 낮아짐.

  ![image](https://user-images.githubusercontent.com/26705935/65689350-dc32fc80-e0a7-11e9-95a2-e6776540bdd6.png)

  - Pre-training에 대해 회의적인 논문에서 주장한 얘기인, *"Random initialized weight도 오래 학습하면 성능이 비슷해진다"* 는 얘기에 대해 반박하는 그림.
  - 만일 학습 데이터가 corrupted 되어있다면, 오래 학습할수록 test error는 증가한다.
  - 따라서, pre-trained weight을 이용해서 *학습 시간을 단축시키는 것은 필수적이다.*

***"Pre-trained weight을 사용한 모델은 Label Corrupted Data에 대해 더욱 강인하다."***

### 2-3. Class Imbalance
- Class가 불균형한 데이터

  ![image](https://user-images.githubusercontent.com/26705935/65689606-5f545280-e0a8-11e9-85de-6ca468999542.png)
  그림: [출처](http://api.ning.com/files/vvHEZw33BGqEUW8aBYm4epYJWOfSeUBPVQAsgz7aWaNe0pmDBsjgggBxsyq*8VU1FdBshuTDdL2-bp2ALs0E-0kpCV5kVdwu/imbdata.png)

  - Instance 수가 많은 **Major 데이터** 와, 수가 적은 **Minor 데이터** 로 구분.
  - 모델 입장에서는 모든 데이터에 대해 "Major class이다" 라고 얘기하면 정확도를 높일 수 있음.
  - Minor 데이터에 대해 정확히 얘기할 수 있도록 모델을 학습시켜야 함.

- Deal with Class Imbalance problem
  - **Oversampling** 과 **undersampling**
    - Oversampling: Minor 데이터의 개수를 증폭함. (ex. SMOTE (*Chawla et al., 2002*))
    - Undersampling: Major 데이터의 개수를 줄임.

  - **Cost sensitive learning** (*Huang et al., 2016*)
    - 학습 과정의 loss function 계산에서, Minor 데이터에 대한 cost를 높게 줌.
    - Minor data의 영향력을 증가시킴.

- 실험 결과

  ![image](https://user-images.githubusercontent.com/26705935/65690129-5dd75a00-e0a9-11e9-9dd7-eb0aab1614d9.png)

  - Class Imbalance 처리 기법 vs. pre-trained 성능 비교.
  - CIFAR-10 및 100 을 class imbalance 하도록 sampling한 데이터를 이용함.
  - Pre-trained weight을 사용하였을 때, imabalanced data를 학습한 모델의 error rate이 다른 Imbalanced data 처리 기법을 적용한 모델에 비해 낮음.

***"Pre-trained weight을 사용한 모델은 Class Imbalanced Data에 대해 더욱 강인하다."***

### 2-4. Out-of-Distribution Detection
- 모델의 Uncertainty에 관한 실험.
- 모델 학습 데이터 (*in-distribution*) 와 다른 분포의 데이터 (*out-of-distribution*) 를 탐지하는 정확도 측정.
  - Out-of-Distribution 설명: [관련 글](https://pod3275.github.io/paper/2019/05/31/OOD.html) 참고.

- OOD Detection method
  - Threshold 기반의 탐지 기법 (*Hendrycks & Gimpel (2017)*)
  - 입력에 대한 모델의 **softmax probability의 최대값** 을 이용하여 OOD를 판단.
    - In-distribution data라면 모델이 높은 확률로 대답할 것임.
    - Out-of-distribution data라면 모델이 낮은 확률로 대답할 것임.

  - Threshold 설정에 따라 AUPR (Area Under Precision Recall) 로 탐지 성능을 측정함.

    ![image](https://user-images.githubusercontent.com/26705935/65691148-4d27e380-e0ab-11e9-998b-602402124a95.png)
    그림: [출처1](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5), [출처2](https://youtu.be/nMAtFhamoRY)

- 실험 결과

  ![image](https://user-images.githubusercontent.com/26705935/65691268-87918080-e0ab-11e9-806d-48bb2b5e8e48.png)

  - 표: 왼쪽의 In-distribution data에 대해, 다양한 OOD data의 탐지 성능.
  - Pre-trained weight을 사용하였을 때, threshold 기반의 OOD 탐지 기법의 성능이 향상함.

***"Pre-trained weight을 사용한 모델은 Out-of-Distribution data를 더욱 잘 구분한다."***

### 2-5. Calibration
- 모델의 Confidence
  - 주어진 입력에 대한 모델의 softmax probability의 최대 값은, **그 모델이 해당 데이터를 이 class로 판단하는 신뢰도** 로 볼 수 있음.
    - 예를 들어, maximum probability가 0.7인 데이터 10개를 모아놓으면, 그 10개 중에 7개가 정답인 것이 자연스러움.

  - 즉, **모델이 내뱉는 확률 = 실제로 맞출 확률** 인 모델을 추구함.

  - 이렇게, 모델이 내뱉는 확률의 최대 값이 모델의 정확도와 같아지는 경우, well-calibrated 되었다고 말함.

- 모델의 Calibration 측정
  - *Root Mean Square Calibration Error (RMS-CE)*

  - $ RMS = \sum_{m=1}^M {\frac{|B_m|}{n} \sqrt {(acc(B_m)-conf(B_m))^2}}$
    - $B_m$ : n개의 데이터 단위 집합.
    - $acc(B_m)$ : $B_m$ 내 데이터에 대한 모델의 정확도.
    - $conf(B_m)$ : $B_m$ 내 데이터의 softmax probability 최대 값의 평균.

  - RMS는 error이기 때문에, 낮을수록 좋음.

- 실험 결과

  ![image](https://user-images.githubusercontent.com/26705935/65692308-4601d500-e0ad-11e9-8333-4393a49a992a.png)

  - Pre-trained weight을 사용하였을 때, 모델의 calibration을 나타내는 RMS error가 감소함.
  - Pre-trained weight을 사용한 모델이 random init weight 모델에 비해 더욱 well-calibrated 됨.

***"Pre-trained weight을 사용한 모델은 well-calibrated 되어있다."***

## 3. Conclusion
- Kaiming He가 주장한 의견인, *"Pre-training은 성능 향상의 효과는 없고, 단지 학습 수렴을 빨라지게 하기 때문에, 꼭 필요하지 않다."* 을 반박함.

- *"Pre-training은 성능 측면이 아닌, 모델의 Robustness 및 Uncertainty 측면에서 향상의 효과가 있다."*
  - Adversarial Robustness, Label Corruption, Class Imbalance
  - Out-of-Distribution Detection, Calibration

- 앞으로의 모델의 robustness 및 uncertainty 향상에 관한 연구는, pre-training을 기본적으로 생각하면서 성능을 평가해야 한다고 주장함.
