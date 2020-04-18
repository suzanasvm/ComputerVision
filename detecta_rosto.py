import cv2
import numpy as np

#Setando o XML de classificacao
def set_classifier(cascPath):
    faceCascade = cv2.CascadeClassifier(cascPath)
    return faceCascade
            
#Comeca a pegar a imagem da WebCam
video_capture = cv2.VideoCapture(0)

while True:
   
   	#Etapa: Aquisicao da imagem
    #Captura e faz a leitura de cada frame em tempo real
    ret, frame = video_capture.read()

    #Etapa: Pre-processamento
    #Espelha a imagem
    frame = cv2.flip(frame,180) 
    #Filtra a imagem em escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Etapa: Deteccao do rosto
    #Seta o classificador do rosto
    faceCascade = set_classifier('classifiers/haarcascade_frontalface_default.xml')

    #Seta as configuracoes do classificador do ROSTO
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(120, 120)
    )

    #Seta o classificador do sorriso
    smileCascade = set_classifier('haarcascade_smile.xml')

    #Desenha um retangulo no rosto encontrado
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
    #Exibe o frame da imagem
    cv2.imshow('Video', gray)

    #Sair do programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
