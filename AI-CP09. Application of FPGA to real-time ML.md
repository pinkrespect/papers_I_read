Application of FPGA to real-time ML

1.  Introduction

    1.  From machine learning to reservoir computing

        1.  Reservoir computing

    2.  Field-Programmable Gate Arrays

        1.  Design flow and implementation tools

2.  Online Training of a photonic reservoir computer

    1.  Introduction

    2.  FPGA Design

    3.  Conclusion

3.  Backpropagation with photonic

    1.  Introduction

    2.  Backpropagation through time

        1.  General idea and new notations

        2.  Setting up the problem

        3.  Output mask gradient

        4.  Input mask gradient

        5.  Multiple input/output

    3.  FPGA Design

    4.  Conclusion

4.  Photonic Reservoir computer with output feedback

    1.  Introduction

    2.  Reservoir computing with output feedback

    3.  Conclusion

5.  Towards online-trained analogue readout layer

    1.  Introduction

    2.  Methods

    3.  Proposed experimental setup

        1.  Analogue Readout Layer

    4.  Conclusion

```{=html}
<!-- -->
```
1.  Introduction

    1.  From machine learning to reservoir computing

        1.  Reservoir computing

-   네트워크 자체를 훈련시키지 않고 단순히 일반적인 선형 판독 계층을
    추가하고 후자만 훈련시키기 위해 반복적인 비선형 네트워크의 역학을
    이용할 수 있음

    -   (판독 가중치를 최적화하기만 하면 되므로) 훈련하기가 상당히 쉬움

-   Reservoir computing

    -   Echo State Network(ESN)

        -   무작위 입력과 내부 연결이 있는 희박하게 연결된 고정 RNNN

        -   일반적으로 Reservoir라고 불리는 은닉층 뉴런은 비선형 활성화
            함수 때문에 입력 신호에 대해 비선형 반응(하이퍼볼릭 접선이
            가장 일반적인 선택)

    -   Liquid State Machine(LSM)

        -   위와 같은 개념이지만, 치솟는 뉴런의 \"수프\"로 구성

        -   \"액체\"라는 명칭은 떨어지는 물체가 만들어내는 액체 표면의
            파동에 비유하여 붙여진 것

    -   이제부터 아이디어를 단순화하기 위해 에코 스테이트 네트워크와
        Reservoir 컴퓨팅을 구분하지 않음

-   Reservoir 컴퓨터 역학

    -   뉴런(노드 또는 저장소의 내부 변수라고도 함) xi

        -   아날로그 뉴런이므로, 별개의 시간 n ∈ Z에서 진화하는 것을
            고려하여, 시간에 따른 변화인 xi(n) 주목

    -   지수 I는 0에서 N -1로 진행

        -   N은 reservoir의 크기 또는 네트워크의 뉴런 개수

    -   N = 50으로 생각(실험에서 흔히 사용되는 값)

![개체이(가) 표시된 사진 자동 생성된
설명](media/image1.png){width="6.261111111111111in"
height="0.7090277777777778in"}

-   여기서 f는 비선형 활성화 함수를 유지

-   u(n)는 시스템에 주입되는 외부 입력 신호

-   aij와 bi는 저장소의 역학을 결정하는 시간 독립 계수

    -   aij는 저장장치 내의 모든 뉴런들 사이의 연결의 강점을 정의하기
        때문에 상호연결 매트릭스라고 불림

        -   0은 가장 강한 연결이고 0은 연결이 없음을 의미

    -   벡터 bi는 입력 중량을 포함

        -   각 뉴런에 대한 입력 강도가 얼마나 강한지 정의

        -   이 계수는 평균이 0인 무작위 분포에서 추출한 값

```{=html}
<!-- -->
```
-   대체적으로 위 방정식은 다음과 같이 표현 가능

![](media/image2.png){width="6.261111111111111in"
height="0.6340277777777777in"}

-   인접 뉴런의 이전 값과 입력 신호를 강조

```{=html}
<!-- -->
```
-   ESN의 개념

    -   \(a\) 매트릭스 aij에 의해 주어지는 뉴런 사이의 연결은 희박해야
        하며(즉, 비교적 낮은 수의 연결로 네트워크 내에 존재 해야 함)

    -   \(2\) 정확한 위상(또는 연결 패턴)은 실제로 중요하지 않음

        -   이는 일반적인 RNN의 관점에서 볼 때 상당한 손실

            -   \"상관 없는\" 이러한 모든 연결은 네트워크를 더 미세하게
                조정하기 위해 대신 훈련될 수 있기 때문

    -   하드웨어 구현의 관점에서 보면 좋음

    -   간단한 토폴로지를 선택하거나 구현에 적합한 특정 토폴로지를
        수동으로 설계 가능

    -   현재의 작업은 Reservoir 컴퓨팅의 광학적 구현에 의존하고 있기
        때문에, 이것은 명심해야 할 중요한 점

    -   그림 I.4와 같이 링 모양의 토폴로지를 사용하는 저장소를 고려

![지도, 텍스트이(가) 표시된 사진 자동 생성된
설명](media/image3.png){width="6.261111111111111in"
height="1.8506944444444444in"}

-   위상 같은 링에 해당하는 가능한 상호 연결 행렬 aij

![전자기기이(가) 표시된 사진 자동 생성된
설명](media/image4.png){width="6.261111111111111in"
height="1.4180555555555556in"}

-   여기서 α는 글로벌 스케일 팩터

-   물리적 체계는 다음과 같이 표현하는 방정식의 집합에 해당

![](media/image5.png){width="6.261111111111111in"
height="0.48541666666666666in"}

-   I.3과의 차이는 노드 x0(n+1)이 연결되는 것과 일치

    -   I.3에서는 xN-1(n)에 연결되고, I.4에서는 xN-1(n-1)에 연결

    -   aij 매트릭스의 구조는 xi-1(n)에 대한 xi(n + 1)의 의존성에 의해
        반영되는 반면, 매트릭스 자체는 단순한 계수 α로 대체

    -   피드백의 강도를 정의, 피드백 이득 또는 피드백 감쇄로 부르기로
        하고, 각각 1보다 크거나 작은지에 따라 달라짐

    -   유사한 방법으로, 우리는 간격에 걸쳐 균일한 분포에서 도출된 전역
        스케일 팩터 β와 벡터 Mi로 바이 벡터를 교체\[-1, +1\]

    -   Mi 벡터는 각 개별 뉴런 xi가 수신하는 입력 신호 u(n)의 강점을
        정의하기 때문에 일반적으로 입력 마스크 또는 입력 가중치라고 불림

        -   따라서 전역 스케일 파라미터 β는 입력 이득이라고 불림

-   비선형 함수 f는 사실상 어떤 경계 함수일 수 있음

    -   현재 심층 학습에 사용되는 소프트맥스 및 하드맥스 함수와 같은
        경계 없는 기능도 작동할 수 있음

        -   일반적인 선택은 쌍곡선 함수 y = tanh(x)를 사용

        -   하드웨어 구현에서 f의 선택은 특정한 비선형 전송 기능을 가진
            장치의 선택에 의해 결정

            -   예를 들어 광학 증폭기의 포화 곡선 또는 포화 소자
                \[34\]일 수 있음

            -   사인 전달 기능이 있는 함수를 사용

-   사인 활성화 함수 f(x) = sin(x)으로 방정식. I.4는 다음과 같이 변함

![](media/image6.png){width="6.261111111111111in"
height="0.48541666666666666in"}

-   이 방정식은 이 작업에서 나중에 논의할 시스템의 행동을 설명

```{=html}
<!-- -->
```
-   네트워크의 출력은 단순한 선형 판독 계층, 즉 저장장치 상태 xi(n)와
    판독 가중치 wi의 선형 조합을 계산하여 얻음

![개체이(가) 표시된 사진 자동 생성된
설명](media/image7.png){width="6.261111111111111in"
height="0.6041666666666666in"}

-   여기서 y(n)는 시간 출력 신호

-   그림 I.4는 방금 설명한 컴퓨터의 그래픽 개요를 보여줌

![지도, 텍스트이(가) 표시된 사진 자동 생성된
설명](media/image3.png){width="6.261111111111111in"
height="1.8506944444444444in"}

-   Reservoir가 하는 일

    -   입력으로 이산 시간 시간 신호 u(n)를 수신하고, 이에 대응하여 또
        다른 이산 시간 시간 신호 y(n)를 생성

    -   무작위 판독 가중치 wi는 출력 신호로, 무엇이든 될 수 있지만
        대부분 쓸모 없는 것

        -   단, 원하는 신호 d(n)로 변환하는 입력 신호 u(n)에 대해 특정
            기능을 수행하는 것이 목표

    -   원하는 출력 d(n)가 입력 u(n)의 몇 가지 값으로 알려져 있다고 가정

    -   예를 들어, u(1 . . . 1000)와 d(1 . . 1000)

    -   이 Time-series들은 정확한 출력을 생성하기 위해 시스템의 판독
        가중치를 조정하는 데 사용될 수 있음. 즉, 우리가 실행하고자 하는
        특정 기능을 모방

    -   일반 RNN을 훈련하려면 모든 내부 연결을 조정하기 위해
        Backpropagation 알고리즘(I.1.2절에서 소개)을 사용해야 한다.

    -   판독 가중치에만 관심을 둘 것

        -   판독값이 선형적이기 때문에, 훈련이 단순해짐

-   훈련 프로세스

    -   목표

        -   u(n)와 d(n)가 모두 알려진 특정 간격 n ∈ \[1,T\] 내에서
            시스템 y(n)와 원하는 출력 d(n) 사이의 차이를 최소화

        -   이 간격은 일반적으로 훈련 간격이라고 하며 그 길이는 시스템을
            최적화하기 위해 얼마나 많은 \"교사\" 입력을 사용했는지를
            정의

        -   d(n)와 y(n) 사이의 거리 측정값 D는 다음을 통해 계산

![개체이(가) 표시된 사진 자동 생성된
설명](media/image8.png){width="6.261111111111111in"
height="0.6194444444444445in"}

-   판독 가중치를 wj로 조정하여 D를 최소화하고 싶으므로, wj에 대한 D의
    파생 모델을 취하면 0

![개체이(가) 표시된 사진 자동 생성된
설명](media/image9.png){width="6.261111111111111in"
height="0.6041666666666666in"}

-   우측을 정리하면, 원하는 출력 d(n)는 wj에 의존하지 않지만 출력 y(n)는
    wj에 의존하며 Eq I.6에 따라 우리는 다음과 같은 식을 구할 수 있음

![](media/image10.png){width="6.261111111111111in"
height="0.4479166666666667in"}

-   Eq I.9를 Eq I.8에 삽입하고 괄호를 확장하면

![](media/image11.png){width="6.261111111111111in" height="1.76875in"}

-   그리고 여기서 선형 방정식의 체계를 얻을 수 있음

![가구이(가) 표시된 사진 자동 생성된
설명](media/image12.png){width="6.261111111111111in"
height="0.2534722222222222in"}

-   Readout 가중치를 적용시키면

![개체이(가) 표시된 사진 자동 생성된
설명](media/image13.png){width="6.261111111111111in"
height="0.5819444444444445in"}

-   Correlation Matrix를 통해 cross-correlation vector를 얻을 수 있음

![개체이(가) 표시된 사진 자동 생성된
설명](media/image14.png){width="6.261111111111111in"
height="0.6041666666666666in"}

-   이 시스템의 솔루션은 아래와 같음

![개체이(가) 표시된 사진 자동 생성된
설명](media/image15.png){width="6.261111111111111in"
height="0.5743055555555555in"}

-   Reservoir 컴퓨터의 훈련은 Correlation 행렬 Rij의 역행으로 귀결

-   미지의 w에 관한 거리 D(Eq I.7)의 최소화는 A가 매트릭스 x이고 b는
    벡터, \|\|은 유클리드 표준인

![가구이(가) 표시된 사진 자동 생성된
설명](media/image16.png){width="6.261111111111111in"
height="0.2986111111111111in"}

-   형식의 문제를 최소화한 것으로 볼 수 있음

-   Eq I.15의 해결은 Ax = b의 선형 시스템을 푸는 것과 같음

![](media/image17.png){width="6.261111111111111in"
height="0.17152777777777778in"}

-   선형 시스템 해결을 위한 표준 접근방식은 최소 제곱 알고리즘을
    사용하여 A를 반전시키는 것

    -   그러나 어떤 경우에는 Ax = b 문제가 잘못 제시된 경우도 있음

    -   즉, 어떤 x도 방정식을 충족하지 못하거나, 둘 이상의 x가 충족하지
        못하거나, 솔루션 x가 매우 큰 값을 가지므로 A나 b의 작은 변동에
        관해서도 불안정 해짐

    -   이 모든 문제는 행렬 A가 작은 또는 사라지는 고유의 값을 가질 때
        발생

-   이러한 경우, 최소 제곱법을 사용하여 과대(과대) 또는 과소(적재)을
    해결

    -   티코노프 정규화 사용

![가구이(가) 표시된 사진 자동 생성된
설명](media/image18.png){width="6.261111111111111in" height="0.26875in"}

-   Γ은 적절히 선택된 티코노프 행렬

    -   고정계수 α로 ID행렬 Γ = αI의 배수로 선택하는 경우가 많음

-   해결책을 정리하면, x = (A + αI)-1b로, 원래의 문제보다 나음

    -   즉, 그러한 정규화는 더 작은 정규화된 값에 우선권

    -   노이즈가 물리적 구현에서 오버핏을 방지하는 역할을 하기 때문에
        시뮬레이션에 주로 사용

    -   일반적으로 reservoir의 크기와 과제에 따라 α ∈ \[10-9, 10-1\]
        설정

1.  Design flow and implementation tools

-   코드

    -   특정 하드웨어 Description 언어(HDL)로 아이디어를 기록(또는 코딩)

        -   VHDL

            -   초고속 통합 회로(VHSIC) HDL

            -   Hard-typed, 긴 구문이 있는 Ada 유사 언어

        -   베릴로그

            -   사용자 친화적인 구문이 있는 Weak-typed C형 언어

    -   마이크로프로세서를 위한 프로그램을 코딩하는 대신에, 사용자는
        특정한 순서로 지시사항을 나열하는 하드웨어 기술 언어를 사용해야
        함

        -   실제로 이 코드는 실행될 의도가 아니라 문자 그대로 전자
            회로의 설계를 기술하는 것이므로 결코 \"프로그램\"이 아님

        -   대부분의 HDL 지침은 \"X의 출력 포트 A를 구성 요소 Y의 입력
            포트 B에 연결\"과 같은 종류

    -   현대 HDL 컴파일러들은 더 복잡한 구조를 산술 연산이나 루프와 같은
        논리로 변환할 수 있다.

        -   HDL 루프는 지침의 루프가 아니며, C 코드처럼 실행되지 않기
            때문에 여기에서 극도로 주의를 기울일 것

-   시뮬레이션

    -   HDL 코드의 구현, 즉 사람이 판독할 수 있는 코드를 FPGA에 로드된
        비트로 변환하는 완전한 프로세스는 매우 복잡한 설계의 경우 최대
        하루까지 오랜 시간이 걸릴 수 있음

        -   따라서, 시뮬렝이션은 논리를 확인할 수 있는 더 나은 선택이 될
            수 있음

        -   다양한 FPGA 에뮬레이션 프로그램으로 시뮬레이션 가능

        -   입력과 클럭 신호를 생성하고, 모든 내부 신호의 시간 흐름에
            따른 동작을 시각화

        -   모든 산술 연산 및 논리 연산이 올바르게 수행되는지 검증하는
            데 유용

-   Synthesis

    -   HDL 코드를 이론 회로로 변환하여 대상 FPGA 칩에서 사용할 수 있는
        표준 구성 요소(원본이라고 함)로 변환

    -   결과는 그림처럼 레지스터-트랜지스터 논리(RTL) 다이어그램으로
        시각화

    -   아직 물리적 칩은 아님

![텍스트, 스크린샷이(가) 표시된 사진 자동 생성된
설명](media/image19.png){width="6.261111111111111in"
height="1.663888888888889in"}

-   Translate

    -   모든 스텝 중 가장 덜 오래 걸리는 스텝

    -   합성된 설계를 입력 및 출력 포트와 결합하고 타이밍과 배치를 해 봄

        -   실제 구현이 이루어지기 전에 모든 설계 파일을 함께 모아보기

-   Map

    -   가장 오래 걸리는 핵심 프로세스

    -   칩의 특수성을 고려하여 이론 설계를 FPGA의 물리적 논리 블록에
        매핑

    -   매핑을 단순화하고 최적화하기 위해 여러 알고리즘을 실행

    -   총 FPGA 리소스 활용도는 프로세스 끝에 표시

        -   예를 들어 CLB 또는 메모리 블록은 설계를 구현하는 데 사용된
            것이 몇 개인지

-   Place and Route

    -   PAR라고도 불림

    -   이 과정은 매핑 과정 중에 이미 행해졌기 때문에 아무 것도 배치하지
        않음

    -   FPGA 라우팅 자원을 이용하여 지도에서 생성되는 구성요소 리스트를
        취하여 서로 연결

    -   작은 디자인에는 쉬운 일이지만, 큰 디자인에는 꽤 도전적이 되고,
        시간이 걸림

    -   칩에 원하는 방식으로 구성요소를 연결하기에 충분한 라우팅 경로가
        없는 경우 매우 큰 설계는 이 단계에서 실패할 수 있음

    -   라우팅 후 PAR은 타이밍 closure을 점검

-   Timing Closure

    -   전기 신호는 FPGA 내에서 유한한 속도로 이동

    -   전파 지연은 매우 짧은 몇 나노초 정도지만, 동기식 설계가 제대로
        작동하려면 고려해야 함

    -   Clock이 빠를수록 이러한 지연은 더욱 심각

    -   시뮬레이션은 이러한 지연을 무시하며 하드웨어 구현 단계에서만
        확인 가능

    -   최대 지연은 사용자가 지정, 일반적으로 \"타이밍 제약 조건\"이라고
        함

    -   실제로 특정 파일에는 칩을 구동하는 데 사용되는 클럭과 주파수가
        나열

    -   이 정보를 통해 PAR은 설계가 이른바 \"Timing Closure\"을
        충족하는지 확인할 수 있음

        -   만일 그렇다면, 그리고 그 논리가 시뮬레이션에서 철저하게
            확인되었다면, 회로는 예상대로 작동해야 함

        -   만약 그렇지 않다면, 그리고 이런 일이 꽤 자주 일어난다면,
            하드웨어를 살피고 소프트웨어를 확인해서 수정해야함

        -   더 복잡한 문제는 느린 시계를 사용하거나 회로를 재설계해야만
            해결할 수 있음

-   비트스트림 생성

    -   마침내 설계가 합성, 배치, 라우팅되고 타이밍 클로징이 충족

    -   마지막 단계는 비트스트림의 생성, 즉 FPGA SRAM에 로드되어 구성을
        설정하는 것

2.  Online Training of a photonic reservoir computer

    1.  Introduction

-   기본 아이디어

    -   광전자 저장장치 컴퓨터에 이 온라인 학습 접근법을 적용하고 그러한
        구현이 실시간 데이터 처리에 매우 적합하다는 것을 보여주는 것

    -   여기서 FPGA 보드의 사용은 불가피

        -   시스템을 실시간으로, 즉 광전자 실험과 병행하여 훈련시켜야 함

        -   시스템은 원칙적으로 어떤 종류의 신호 처리 작업, 특히 시간에
            따라 달라지는 작업에 적용될 수 있음

-   FPGA 회로에의 적용

    -   온라인 상태에서 광전자 저장 장치를 사용하는 컴퓨터를 사용하기
        위해서는 FPGA 회로의 사용이 필수적

    -   주된 병목현상은 왜곡된 신호를 정확하게 샘플링하기 위해 존재하는
        아날로그-디지털 컨버터(ADC)에 있음

    1.  Equalisation of non-stationary channels

        1.  Influence of channel model parameters on equaliser
            performance.

-   Eqs I.18과 I.19는 채택된 계수의 숫자 값으로 정의되는 일정한 양의
    기호 간섭과 비선형 왜곡을 가진 특정 채널을 모델링

    -   이 미립자 채널 모델을 보다 잘 이해하고, 어떤 입력 신호 왜곡
        단계가 동등하기 가장 어려운지를 보여주기 위해 보다 일반적인 채널
        모델을 도입

![](media/image20.png){width="6.261111111111111in"
height="1.6340277777777779in"}

-   위의 식에 의해 주어진 보다 일반적인 채널 모델을 도입, 파라메터 pi와
    m의 다른 값에 대한 평준화 성능을 조사

-   채널 임펄스 응답의 일반적인 모양을 유지하기 위해 우리는 Eq II.1의
    1에 d(n) 계수를 고정

-   그림 II.1은 몇 가지 m 값에 대해 Eq II.1에 의한 결과

![스크린샷이(가) 표시된 사진 자동 생성된
설명](media/image21.png){width="6.261111111111111in"
height="3.9854166666666666in"}

1.  Slowly drifting channel.

-   여태 주어진 모델은 이상주의적인 정지 잡음 무선 통신 채널 즉, 채널이
    전송 중에 동일하게 유지되는 것을 설명

-   무선통신에서는 수신 신호가 큰 영향을 미침

-   매우 가변적인 성격으로 볼 때, 채널의 속성은 실시간으로 중요한 변화에
    노출될 수 있음

-   이 시나리오를 조사하기 위해, 신호 전송 중에 파라미터 pi 또는 m이
    실시간으로 변화하는 \"drafting\" 채널 모델로 일련의 실험을 수행

    -   이러한 변화는 저속으로 발생했는데, Reservoir 컴퓨터를 훈련시키는
        데 필요한 시간보다 훨씬 느림

    -   우리는 두 고정 값 사이의 단조로운 증가(또는 감소)와 느린
        진동이라는 두 가지 변화 패턴을 연구

        1.  Switching channel.

-   채널 모델이 즉시 전환되는 "스위치" 채널

    -   채널 property은 slowly drifting parameter 외에도 급격한 환경
        변화로 인해 급격한 변동에 노출될 수 있음

    -   더 나은 실용적 평준화 성능을 위해 중요한 채널 변화를 감지하고 RC
        판독 가중치를 실시간으로 조정할 수 있어야 함

    -   Reservoir 컴퓨터는 그러한 변화를 감지하고 새로운 훈련 단계를
        자동으로 트리거하여 판독 가중치가 새로운 채널의 평준화에 맞게
        조정되도록 해야 함

    -   구체적으로는 Eqs I.18과 I.19에 의해 주어진 상수 채널 대신에
        비선형도

![](media/image22.png){width="6.261111111111111in"
height="0.8506944444444444in"}

> 에서 위와 같은 다른 3개의 채널을 도입하고, 수식을 변경하지 않고 한
> 채널에서 다른 채널로 정기적으로 전환

1.  FPGA Design

![텍스트이(가) 표시된 사진 자동 생성된
설명](media/image23.png){width="6.261111111111111in"
height="4.298611111111111in"}

2.  Conclusion

-   이번 실험에서 한 일

    -   FPGA 칩에 Simple Gradient Descent 알고리즘을 프로그래밍

    -   Non-linear Channel Equalisation 과제에서 시스템을 테스트

-   결과

    -   Channel Equalisation 과제에서 이전보다 최대 두 자릿수 낮은
        오류율을 얻음

    -   실험 런타임을 크게 줄임

-   Drifting 채널과 Switching 채널을 동등하게 함으로써, Non-stationary
    작업에 적합하다는 것을 증명

-   설정으로 가능한 한 가장 낮은 오류율을 얻음

3.  Backpropagation with photonic

    1.  Introduction

-   백프로파게이션(BP) 알고리즘\[82, 83\]

    -   BP 알고리즘의 아이디어는 시스템의 매개변수 공간에서 비용 함수의
        파생(또는 기울기)을 계산

    -   비용 함수를 줄이기 위해 매개변수 자체에서 gradient를 뺌

    -   이 과정은 비용 기능이 더 이상 감소하지 않을 때까지 반복

-   하드웨어 시스템에서 BP algo-rithm을 구현하는 것은 기울기를 계산하는
    정확한 모델의 필요와 BP 알고리즘 실행에 필요한 자원 때문에 어려울 수
    있음

    -   특정 경우에 최적화되 시스템에 물리적으로 구현

    -   자체 학습 컴퓨팅 시스템은 처리 속도 또는 제한된 전력 소비
        측면에서 얻는 이득도 훈련 단계에도 적용될 것이기 때문에 매우
        유리

    -   또한 BP 알고리즘이 동일한 하드웨어 컴퓨팅을 갖는 것은 시스템의
        정확한 모델 필요성을 상당 부분 제거

-   광전자 Reservoir 컴퓨터에 BP 알고리즘을 물리적으로 구현

    -   핵심은 비선형성과 그 파생성을 모두 구현할 수 있는 광학적 요소를
        추가하여 시스템을 신호 처리기로 사용하고 BP 알고리즘을 수행할 수
        있도록 수정하는 것

    -   BP 알고리즘을 사용할 때 기술 결과의 상태를 파악하기 위해
        Real-world 음성 인식 과제를 포함하여 기계 학습 커뮤니티에서
        어렵게 고려된 몇 가지 작업에 대해 시스템을 테스트

    1.  Backpropagation through time

-   backpropagement는 반복되는 신경망을 훈련하기 위한 Time-honoured 방식

    -   본질적으로, 원하는 네트워크 출력에 기초하여 비용 함수를
        정의하고, 체인 규칙을 사용하여 시스템의 내부 매개변수(가중치)에
        관하여 이 비용 함수의 기울기를 결정

    -   용어 "Backpropagation\"는 RNN의 재귀로 인해 Gradient을 계산하는
        것은 시스템 업데이트를 통해 오류 신호를 역방향으로 전파하는 것을
        수반한다는 사실에서 유래

-   알고리즘 작동 방식

    -   RNN은 입력에서 출력 뉴런까지 일정 수의 타임스텝(업데이트)으로
        네트워크를 통해 전파되는 입력 I를 공급

    -   판독 가중치 세트를 사용하여 출력 O를 계산

    -   목표 출력 T와 비교하고 오류 E를 계산

    -   이 시점에서 네트워크는 반전

    -   오류 신호 E는 전방으로 전파된 초기 입력 신호와 동일한 방식으로
        출력에서 입력 뉴런까지 선형화된 네트워크를 통해 뒤로 전파됨

        -   즉, 출력 뉴런은 입력으로 작용하고 입력 뉴런은 시스템
            출력으로 간주

        -   이것은 각 뉴런의 오차를 계산할 수 있어야 함

        -   즉, 원하는 값 T에 더 가까이 다가가기 위해 그 값이 어떻게
            바뀌어야 하는지를 계산할 수 있음

    -   분명히 뉴런 상태는 \"수정\"될 수 없지만, 가중치는 될 수 있음

        -   따라서 이러한 오류는 입력, 내부 및 판독 가중치에 대한 보정을
            계산하여 주어진 입력에 대한 시스템 출력 O가 목표 T에
            근접하도록 함

    -   O와 T 사이의 오차를 더 이상 줄일 수 없을 때까지 (일반적으로
        작은) 보정은 반복적으로 적용

        1.  General idea and new notations

-   일반적인 RC 태스크에서 목표는 입력 시퀀스 si(여기서 i ∈ {1, ..., L},
    총 시퀀스 길이 L)를 출력 시퀀스 yi에 매핑하는 것

-   출력 시퀀스 yi는 예를 들어 음성 신호와 같은 표적 값을 가지고 있음

-   Delay-coupled 시스템을 Reservoir 컴퓨터로 사용하기 위해, 이산 시간
    입력 시퀀스 si는 입력 마스크 m(r)와 바이어스 마스크 mb(r)에 의해
    연속 시간 함수 z(t)로 인코딩

-   여기서 r은 다음과 같이 T는 마스킹 기간을 갖게 됨

![](media/image24.png){width="6.261111111111111in"
height="0.2986111111111111in"}

-   사인 비선형성을 가진 광전자 저장기 컴퓨터는 방정식

![](media/image25.png){width="6.261111111111111in" height="0.23125in"}

> 을 준수하며 여기서 a(t)는 상태 변수, D는 지연

-   인자 μ는 총 루프 증폭에 해당

-   Eq III.2는 이케다 지연 미분 방정식의 특수한 경우로 볼 수 있음

-   그런 다음 연속 시간 상태 변수 a(t)를 이산 시간 출력 시퀀스 yi에 매핑

-   r ![](media/image26.emf) \[0,T\]와 다음과 같이 편향 ub를 사용하는
    출력 마스크 u(r)를 사용

![](media/image27.png){width="6.261111111111111in"
height="0.49236111111111114in"}

-   RC 패러다임에서 입력 마스크는 일반적으로 무작위로 선택

-   출력 마스크 u(r)와 ub는 원하는 출력
    ![](media/image28.png){width="1.0819444444444444in"
    height="0.2013888888888889in"}사이의 평균 제곱 오차 C를 최소화하는
    방정식의 선형 시스템을 풀어서 결정

-   RC에 에러 백프로파게이션를 적용하는 목적은 저장장치 상태 a(t)를 알고
    입력 및 출력 마스크 m(r), mb(r), u(r) 및 ub를 최적화하는 것

![](media/image29.png){width="6.26875in" height="1.836111111111111in"}

-   여기서 제공하는 마스크에 대한 오류 함수
    ![](media/image28.png){width="1.0819444444444444in"
    height="0.2013888888888889in"}의 기울기가 필요

-   여기서 e(t) = ∂C/∂a(t)는 연속 시간 신호이며, 위와 같이 i ∈ {1, · · ·
    , L}과 r ∈ \[0,T\]에서 주어짐

-   C를 낮추기 위해 마스크를 반복적으로 개선 가능

    1.  FPGA Design

![스크린샷, 텍스트이(가) 표시된 사진 자동 생성된
설명](media/image30.png){width="6.26875in" height="4.163888888888889in"}

2.  Conclusion

-   입력 마스크와 출력 마스크를 모두 최적화함으로써 지연 기반 저장장치
    컴퓨터의 성능을 획기적으로 개선

-   기반 하드웨어가 자체 최적화 프로세스의 많은 부분을 실행할 수 있음

-   우리는 빠른 전자 광학 시스템과 RC 커뮤니티에서 어렵게 간주되는
    작업에서 데모를 실시

-   세 가지 과제 모두에서 얻은 성능 이득이 수치 시뮬레이션에서 예측한
    것과 유사했기 때문에, 우리의 연구는 BP 알고리즘이 다양한 실험 결함에
    대해 강력하다는 것을 밝혀냄

-   사인 비선형성과 그것의 코사인 분리성에 의존하지만, 다른 비선형
    기능도 그 파생물과 함께 하드웨어에서 성공적으로 실현될 수 있음

    -   예를 들어, 입력 신호를 특정 임계값 이하로 절단하는 이른바 linear
        rectifier 함수는 신경 구조에서 널리 사용되는 활성화 함수이며,
        아날로그 스위치를 사용하여 쉽게 구현할 수 있는 단순한 이진함수

    -   sigmoid 비선형성과 그 파생성을 구현 가능

    -   BP 알고리즘의 물리적 구현이 매우 다양한 물리적 시스템에서 가능할
        것으로 예상한다.

-   설정은 여전히 gradient를 계산하고 마스킹을 수행하기 위해 느린 디지털
    처리가 필요

-   아날로그 하드웨어에서 마스킹 작업을 수행하는 것은 활발하게 연구 중

-   FPGA에 전체 훈련 알고리즘을 구현하면 실험 속도가 급격히 증가할 것

4.  Photonic Reservoir computer with output feedback

    1.  Introduction

-   Reservoir 컴퓨팅은 few future timesteps를 생성하는 데 초점을 맞춘
    단기 예측 작업에 직접 적용할 수 있음

    -   롱 호라이즌 예측에 대해서는, Reservoir로 RC 출력 신호를 다시
        공급함으로써 가능

    -   이 추가 피드백은 시스템의 내부 역학을 상당히 활성화하여 입력
        신호를 수신하지 않고 자동으로 생성할 수 있음

    -   Reservoir 컴퓨팅은 Chaotic Series의 장기 예측에 사용 가능

    -   출력 피드백이 있는 Reservoir 컴퓨터는 주기적 신호 생성과 조정
        가능한 주파수 생성의 더 쉬운 작업을 수행

-   Reservoir 컴퓨팅은 생물학적으로 영감을 받은 알고리즘

    -   실제로 신피질 논문들의 주요 동기 중 하나는 신피질 내의 마이크로
        회로가 정보를 어떻게 처리할 수 있는지를 제안하는 것

        -   특정 속성을 가진 Time-series 생성은 생물학적 신경 및 화학적
            회로(예: 이동 제어, 생물학적 리듬 등)의 중요한 특성

        -   훈련 가능한 Time-series을 생성하기 위해 Reservoir 컴퓨터와
            유사한 생물 회로가 사용되었는지를 실험적으로 조사

    -   특정 속성을 가진 Time-series 생성은 신호 생성과 처리에서 중요한
        작업

        -   광 저장장치 컴퓨팅이 초고속 및 저에너지 광학 신호 처리를
            수행할 수 있는 가능성을 고려할 때, 이는 실험의 결함에 대한
            태스크의 이해를 목적으로 탐색해야 하는 영역

        -   Chaotic Time-series을 모방하는 시스템을 고려할 때,
            에뮬레이션의 품질 정량화

            -   물리적 시스템이 소음의 영향을 받아 대상 혼돈 시간
                시리즈의 대략적이고 노이즈가 낀 에뮬레이션보다 더 나은
                결과를 산출할 수 없기 때문에 이 질문에 대한 대답은 실험
                구현의 경우에 필수적

            -   새로운 평가 지표 개발 필요

-   원칙적으로 출력 신호를 실시간으로 생성 및 공급할 수 있는 빠른 판독
    계층이 필요

    -   FPGA 칩에 구현된 빠른 실시간 디지털 판독 계층의 접근 방식을 선택

    -   Opto-electronic 시스템을 Reservoir로 사용

-   주파수 스펙트럼 비교 및 무작위 시험과 같은 몇 가지 새로운 접근법을
    소개

    -   이러한 접근방식은 잘 알려진 신호 분석 기법에 기초하지만,
        Reservoir 컴퓨터에서 발생하는 혼란스러운 신호의 평가를 위해 채택

    -   RC가 목표 궤적을 따르는데 어려움을 겪지만, 출력이 목표
        time-series의 핵심 특성을 정확하게 재현

    1.  Reservoir computing with output feedback

-   RC는 이제 입력으로 두 개의 다른 신호를 수신할 수 있기 때문에, 우리는
    입력 신호 I(n) = u(n)일 수 있는 입력 신호를 I(n)로 표시해야 하며,
    입력 신호는 하나의 타이밍 I(n) = y(n - 1)로 지연될 수 있음

-   reservoir 컴퓨터는 그림에서 묘사된 두 단계로 운영

![텍스트, 지도이(가) 표시된 사진 자동 생성된
설명](media/image31.png){width="6.26875in" height="4.163888888888889in"}

-   훈련 단계 및 자율 주행. 훈련 단계에서 탱크 컴퓨터는 시간 다중화 교사
    신호 I(n) = u(n)에 의해 구동되며, 내부 변수 xi(n)의 결과 상태를 기록

    -   Teacher signal는 조사 중인 과제(IV.3절에 소개될 것)에 따라
        달라짐

    -   이 시스템은 현재에서 교사 시간 시리즈의 다음 값을 예측하도록
        트레이닝

        -   즉, 판독 가중치 wi는 가능한 한 y(n) = u(n + 1)에 가깝게
            최적화

        -   저장장치 입력이 Teacher 시퀀스에서 저장장치 출력 신호 I(n) =
            y(n - 1)로 전환되고 시스템이 자동으로 작동

        -   저장장치 출력 y(n)는 실험의 성능을 평가하는 데 사용

    1.  Conclusion

-   하드웨어 저장장치 컴퓨팅에서 출력 피드백의 잠재력을 입증하며
    반복되는 신경망의 자율적인 광학 구현을 위한 중요한 단계를 구성

    -   구체적으로는 다른 주파수의 사인파와 짧은 무작위 패턴을 모두
        상당한 안정성으로 생성할 수 있는 광학 저장기 컴퓨터를 제시

    -   유사한 주파수 스펙트럼과 상당히 가까운 Lyapunov 지수인
        Mackey-Glass 시간 시리즈를 모방

    -   로렌츠 시스템의 역학을 효율적으로 포착, 무작위 유사 특성을 가진
        time-series을 생성

-   Reservoir 컴퓨터의 판독은 빠른 FPGA 칩으로 실시간으로 수행

    -   이것은 아날로그 장치에 디지털 출력 계층을 생성

        -   아날로그 피드백을 가지고 시스템에 영향을 미칠 많은 문제들을
            조사할 수 있음

    -   출력 피드백에 아날로그 피드백을 사용하면 입력 마스크에 의한 샘플
        및 홀드 회로, 증폭 및 곱셈으로 구성된 추가 전자 회로를 추가할
        필요가 있음

-   노이즈에 대해서

    -   대책이 없음

    -   소음이 덜하고 새로운 구성 요소로 재구축

    -   SNR을 60dB 이상으로 증가시키기는 어려울 것

5.  Towards online-trained analogue readout layer

    1.  Introduction

-   Reservoir 컴퓨팅의 실험 구현의 주요 단점

    -   효율적인 판독 메커니즘의 부재 즉, 뉴런의 상태를 수집하고
        컴퓨터에서 후 처리하여 처리 속도를 심각하게 감소시켜 적용성을
        제한

        -   아날로그 판독으로 이 문제를 해결 가능

        -   이 연구 방향은 이미 읽기 계층의 복잡한 구조로 인해 상당한
            성능 저하

        -   실제로 이러한 작업에 사용된 접근방식은 선형 출력 계층을 높은
            정확도로 특징짓는 것이었고, 그 결과 출력 가중치를
            오프라인으로 계산 가능

        -   그러나 설정의 각 하드웨어 구성요소를 충분한 정확도로 특징
            짓는 것은 사실상 불가능

        -   또한 출력 계층의 구성요소는 약간의 비선형 동작을 가질 수
            있음

        -   따라서 이 접근방식은 만족스럽게 작동하지 않음

-   온라인 학습 접근법으로 단점 해결

    -   온라인 교육은 입력 데이터를 이용할 수 있게 되면서 시스템을 점차
        최적화할 수 있게 하기 때문에 기계 학습 커뮤니티에서 많은 관심

    -   온라인 접근방식은 입력의 변동에 따라 모델을 업데이트할 수 있기
        때문에 시간이 지남에 따라 특성이 변하는 비정기 입력 신호에도
        쉽게 대처할 수 있음

    -   하드웨어 시스템의 경우 시스템이 하드웨어 구성요소의 점진적인
        변화에 적응하기 때문에 온라인 교육은 하드웨어의 변동에 쉽게 대처
        가능

-   Reservoir 컴퓨팅의 맥락에서, 온라인 훈련은 경사도 하강을 실행

    -   점차적으로 작업에 적응하기 위해 출력 층을 바꿔, 일련의
        매개변수(판독 가중치)에 의해 특성화

    -   온라인 훈련에서는 이러한 가중치를 조금씩 조정하여 시스템의
        출력이 목표 신호에 가까워지도록 함

    -   현재의 맥락에서 중요한 점은 이전에 사용되었던 오프라인 방법과
        비교하여, 경사 강하 기반의 온라인 교육에서 이러한 가중치가 출력
        신호에 어떻게 기여하는지에 대한 가정이 필요하지 않다는 것

        -   즉, 출력 계층을 모델링할 필요는 없음

    -   더욱이, 판독 계층의 전송 기능은 원칙적으로 비선형일 수 있음

        -   실제 숫자 시뮬레이션을 사용하여 이러한 기능이 하드웨어
            저장장치 컴퓨터를 훈련시키는 데 어떻게 매우 유리할 수 있는지
            보여 줌

-   정확성

    -   FPGA 칩이 간단한 그라데이션 강하 알고리즘을 실시간으로 처리하여
        온라인에서 교육하는 아날로그 계층을 추가

    -   판독 계층은 \[73\]에서 출력 신호의 진폭을 증가시키는 데 사용된
        더 복잡한 RLC 회로(저항기 R, 인덕터 L 및 콘덴서 C로 구성됨)
        대신, 저항기-용량기(RC) 회로로 구성

-   이 세팅의 성능

    -   온라인 교육 접근법에 의해 거의 완전히 완화

    -   상대적으로 간단한 아날로그 판독 계층을 사용하여 온라인에서
        교육을 받고 기본 프로세스를 모델링하지 않음

    -   오프라인에서 훈련된 디지털 계층에서 생산된 것과 유사한 결과를
        얻음

    -   비선형 판독 기능을 가진 특별한 경우를 탐구하고 이 복잡성이
        시스템의 성능을 크게 감소시키지 않는다는 것을 확인

    1.  Proposed experimental setup

        1.  Analogue Readout Layer

-   아날로그 판독 계층

    -   Reservoir로부터 받는 광학 전력은 두 개로 나누어짐

        -   판독 광다이오드(TTI TIA-525I)로 전송

            -   Reservoir 상태 xi(n)를 포함하는 결과 전압 신호는
                트레이닝 과정을 위해 FPGA에 의해 기록

        -   FPGA의 DAC에서 생성된 판독 가중치 wi를 적용하는 이중 출력
            마하-Zehnder 변조기(EOSPACE AX-2X2-0MSS12)에 의해 변조

            -   변조기의 출력은 두 입력에서 받은 광도의 차이에 비례하는
                전압 레벨을 생성하는 균형 광도계(TTI TIA-527)에 연결

            -   이를 통해 Reservoir의 상태를 양의 중량과 음의 중량으로
                곱할 수 있음

            -   가중 상태의 합계는 저역 통과 RC 필터에 의해 수행

            -   도면에 표시되지 않는 필터의 저항 R은 균형 잡힌
                광다이오드의 출력 임피던스 50Ω

            -   결과 출력 신호는 y(n)에 비례하여 FPGA가 훈련 및 성능
                평가를 위해 기록할 때도 있음

-   아날로그 판독 계층 출력의 명시적 계산

    -   콘덴서는 균형 잡힌 광다이오드의 출력을 지수 커널과 시간 상수 τ와
        통합한다. RC 필터의 임펄스 응답은

![](media/image32.png){width="6.26875in" height="0.42569444444444443in"}

-   에 제시되어 있고, 콘덴서의 전압 Q(t)는

![](media/image33.png){width="6.26875in" height="0.5298611111111111in"}

-   에 의해 주어지며, 여기서 X(t)는 저장장치 상태를 포함하는 연속
    신호이고 W(t)는 이중 출력 강도 변조기에 적용되는 판독 가중치

-   출력 y(n)는 별도의 시간 t = nT :

![](media/image34.png){width="6.26875in" height="0.2013888888888889in"}

-   에서 캐패시터의 전하에 의해 주어짐

-   X(t)와 W(t)는 부분 함수 X(t) = xi(n) 및 W(t) = wi for t t ∈ \[θ(i -
    1), θi\]이고 여기서 duration = T/N은 하나의 뉴런의 지속시간이기
    때문에, 우리는

![](media/image35.png){width="6.26875in" height="1.2166666666666666in"}

-   을 얻기 위한 이산 집계에 의한 통합의 근사치 확인 가능

-   판독 계층 출력 y(t) = Q(t)는 따라서 의 선형 조합이다.

-   저장소는 xi로, wi와 RC 필터의 지수 커널에 의해 가중치를 부여

    -   일반적인 Reservoir 컴퓨터 출력과는 달리 Eq V.4에서 n의 출력은
        현재 상태 xi(n)뿐만 아니라 이전 시간 xi(n - k)의 상태에 따라
        달라짐

-   아날로그 판독값의 이전 실험조사에서 판독 가중치 wi는 Eq I.6에 의해
    주어진 출력 신호를 가정하여 Reidge Regression (I.1.3항 참조)를
    사용하여 계산

-   그러나 실험이 대신 Eq V.4와 유사한 출력을 생성했기 때문에 판독
    가중치를 적절하게 수정

-   가중치 wi는 저장장치 출력 신호 y(n)와 목표 출력 d(n)를 일치시키도록
    점진적으로 조정

-   이러한 가중치가 출력 신호 y(n)에 실제로 어떻게 기여하는지에 대한
    가정은 없음

1.  Conclusion

-   광 저장기 컴퓨터의 아날로그 판독 계층 성능 향상을 위한 온라인 학습
    기법 제안

    -   간단한 RC 필터를 기반으로 한 출력 레이어로 광전자 설정을
        시연하고 두 가지 벤치마크 작업에 대해 수치 시뮬레이션을 사용하여
        테스트

    -   간단한 Gradient Descent 알고리즘을 사용하여 온라인으로
        훈련시키면, 디지털 판독 계층과 동일한 수준의 성능을 얻을 수 있음

    -   기본 하드웨어의 모델링이 필요하지 않으며, 매개변수 또는
        구성요소의 부정확한 선택과 같은 가능한 실험 결함에 대해 강력한
        결과

    -   광다이오드(photodiodes)의 saturable response 같은 판독 계층의
        비선형성을 처리 가능

    -   온라인 훈련의 도입이 모든 하드웨어 저장장치 컴퓨터에 적용 가능

-   효율적인 아날로그 판독 계층을 실현하면

    -   Fully-Analogue 고성능 RC를 구축

    -   느린 디지털 후처리를 포기

    -   빠른 광학 구성요소를 최대한 활용 가능
