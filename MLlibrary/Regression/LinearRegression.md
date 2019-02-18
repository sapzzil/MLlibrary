# Linear Regression
## Contents
* Linear Regression?
* Cost Function
* Gradient Descent

## Linear Regression?
![Mean Squared Error (MSE)](MLlibrary/README/img/Linear Regression.JPG)
* 선형회귀
* 어떠한 현상에 대한 데이터값이 선형적인 상관관계를 보이면 이를 대표하는 모델을 구하는 방법
* 모델의 형태는 y = wx + b
* 학습이라는 과정은 x(독립변수,실험변수)와 y(종속변수)를 만족시킬 수 있는 w(weight, coefficient)와 b(bias, intercept)를 구하는 과정
* 학습과정에서 w와 b를 구하는 과정은 실제 y값과 모델을 통해 구한 y값의 비교를 통해 그 오차가 최소가 되게 하여 구한다 (손실함수의 최소화)
* 경사하강법을 통해 손실함수를 최소화 할 수 있는 w와 b를 구한다

## Cost Function
* 손실함수
* 다양한 손실함수 존재
* 여기선 MSE(Mean Squared Error)를 다룸
* 오차 제곱의 합을 나타내고 0에 가까울 수록 모델과 실제의 차이가 적음을 의미한다


## Gradient Descent
* 경사하강법
* Cost/loss Function에서 각 독립변수(Weight와 bias)에 대한 편미분을 사용
* 초기의 w와 b에 대한 기울기에 학습률(Learning Rate)를 곱하여 w와 b 에서 뺀다(ex> w – learning rate * w기울기)
* 여기서 미분값을 빼는 이유는 사용한 cost/loss function이 Mean Squared Error이고 이는 제곱이기에 아래로 볼록한 이차곡선을 그리게 된다. 이차곡선에서 가장 cost가 작은 값은 기울기가 0이고 어떤 지점에서 최소값을 찾아가려면 기울기와 반대방향인 음의 방향으로 움직여야 하기에 뺄셈을 하게 된다.

