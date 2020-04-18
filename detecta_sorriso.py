import cv2
import numpy as np

#Setando o XML de classificacao
def set_classifier(cascPath):
    cascPath = cascPath
    faceCascade = cv2.CascadeClassifier(cascPath)
    return faceCascade

def detectSmile(img,faces):
    
    	 for (x, y, w, h) in faces:
            smileCascade =  set_classifier('classifiers/haarcascade_smile.xml')
            roi_smile = img[y:y+h, x:x+w]
            smile = smileCascade.detectMultiScale(
                roi_smile,
                scaleFactor=1.1,
                minNeighbors=70,
                minSize=(15, 15)
                )
            #Create the Smile Rectangle
            for (sp, sq, sr, ss) in smile:         
                cv2.rectangle(roi_smile,(sp,sq),(sp+sr,sq+ss), (0,0,255),3)
                return True
            return False 
            
def put_msg(msg,frame,px,py):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0,0,255)
    cv2.putText(frame,msg,(px,py), font, 1.5,(0,0,0),3)


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
        frame,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(120, 120)
    )

    #Desenha um retangulo no rosto encontrado
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        #Etapa: Deteccao do sorriso
        #Detecta sorriso
        smile = detectSmile(frame,faces)
        #Se o sorriso for detectado
        if(smile):
        	put_msg("Smile detected",frame,50,50)

    #Exibe o frame da imagem
    cv2.imshow('Video', frame)

    #Sair do programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
