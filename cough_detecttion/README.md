# Cough Detection

기침 소리를 감지하는 시스템  
소리를 Mel spectrogram 이미지로 변환하여 기침인지 아닌지 분류하는 방법 사용  
  
사용한 데이터셋 : ESC-50 (https://github.com/karolpiczak/ESC-50)
사용한 모델 : EfficientNet

추가로 연구할 점  
데이터셋의 구조가 50클래스, 클래스별 샘플이 40개로 기침 데이터가 부족함  
이를 극복하기 위해서 별도의 녹음을 통한 기침 데이터셋 확보가 필요함  

  
1. melspectrogram_conver.py -> audio 파일을 png파일로 바꿔주는 코드
2. train.py -> 모델을 학습시키는 코드
"# cough_detection" 
