import cv2
import os
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer




#telechargement des clasifieurs qui detecte le visage et les yeux
visage = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_default.xml')
goeil = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
doeil = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')


#telechargement des poid du model CNN
model = load_model('models/model.h5')

#La police utilisée pour l'écriture
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

score=0
#initialisation avec un tableux vide 
droit_pred=[99]
gauch_pred=[99]
#Téléchargement du son de l'alarme
mixer.init()
sound = mixer.Sound('telsig1.wav')
#Pour capturer une vidéo apartire du camera de notre pc
cap = cv2.VideoCapture(0)
while(True):
    # lire la capture du vidéo
    ret, frame = cap.read()
    #vérification si la cameraest accessible
    if not ret:
        print("Impossible de recevoir la trame (fin du flux?). Quitter...")
        break
     
    # la taille de l'image capture apartire du camera 
    hauteur,largeur = frame.shape[:2]
    
    #changer couleur du frame du RGB to niveau de gris pour minimisé la compléxité du programme
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #recherche des coordonnées du visage, oeil droit et gauche 
    visages = visage.detectMultiScale(gray)
    gauche_oeil = goeil.detectMultiScale(gray)
    droite_oeil =  doeil.detectMultiScale(gray)

    # desiné un rectangle au tours du visage
    for (x,y,w,h) in visages:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 2 )
        break
    
    
    for (x,y,w,h) in droite_oeil:
        #cropé la partie dont il ya l'oeil droit
        d_oeil=gray[y:y+h,x:x+w]
        #changer les dimensions de l'image
        #La méthode resize() modifie directement le tableau d'origine.
        d_oeil = cv2.resize(d_oeil,(24,24))
        #normalisation de l'image
        d_oeil= d_oeil/255
        #La méthode resize() ne modifie pas le tableau d'origine.
        d_oeil=  d_oeil.reshape(24,24,-1)
        #Cette fonction étend le tableau en insérant un nouvel axe à la position spécifiée.
        d_oeil = np.expand_dims(d_oeil,axis=0)
        #passe l'image traiter au classifieur cnn
        droit_pred = model.predict_classes(d_oeil)
        if(droit_pred[0]==1):
            lbl='Open' 
        if(droit_pred[0]==0):
            lbl='Closed'
        break
       

    for (x,y,w,h) in gauche_oeil:
        g_oeil=gray[y:y+h,x:x+w]
        g_oeil = cv2.resize(g_oeil,(24,24))
        g_oeil= g_oeil/255
        g_oeil=g_oeil.reshape(24,24,-1)
        g_oeil = np.expand_dims(g_oeil,axis=0)
        gauch_pred = model.predict_classes(g_oeil)
        if(gauch_pred[0]==1):
            lbl='Open'   
        if(gauch_pred[0]==0):
            lbl='Closed'
        break
    # si les les deus yeux sans fermé 
    if(droit_pred[0]==0 and gauch_pred[0]==0):
        #incrémentation du score
        score=score+1
        #ecrire l'état des yeux et le score dans frame
        cv2.putText(frame,"Closed",(10,hauteur-20), font, 1,(255,255,255))
        cv2.putText(frame,'Score:'+str(score),(100,hauteur-20), font, 1,(255,255,255))
    else:
        if(droit_pred[0]==1 or gauch_pred[0]==1):
            #remetre le score a 0 
            score=0
            #ecrire l'état des yeux et le score dans frame
            cv2.putText(frame,"Open",(10,hauteur-20), font, 1,(255,255,255))
            cv2.putText(frame,'Score:'+str(score),(100,hauteur-20), font, 1,(255,255,255))
        else:
            #si aucune visage est devant la caméra
            score=0
            cv2.putText(frame,"NO FACE DETECTED",(10,hauteur-20), font, 1,(255,255,255))
  
    #10 est choisis par test 
    if(score>10):
        #la personne a sommeil alors nous sonnons l'alarme
        # try pour verifier si en peux jouer à l'alarme
        try:
            #jouer à l'alarme
            sound.play()
        except:  
            pass
    #affichage du frame
    cv2.imshow('frame',frame)
    #initialisation des valeurs de classification pour oublié les valeurs actuelle 
    droit_pred[0]=2
    gauch_pred[0]=2
    #verification si l'utilisateur tappe "q" pour quitté
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#Libération du camera
cap.release()
#fermétous les fenetres
cv2.destroyAllWindows()