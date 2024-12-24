import cv2
import mediapipe as mp
import time

öncekiZaman = 0
suankiZaman = 0

cap = cv2.VideoCapture(0)

# MediaPipe El Algılama Modülünü Başlatma
mpEl = mp.solutions.hands  # MediaPipe kütüphanesindeki el algılama modülünü başlatır.
eller = mpEl.Hands(max_num_hands=2)
mpCizim = mp.solutions.drawing_utils # Tespit edilen el eklemlerini ve bağlantılarını çizmek için kullanılır.


while True:
    basari,img = cap.read()
# basari: Görüntünün başarıyla okunup okunmadığını kontrol eder.
# img: Okunan görüntüyü tutar.
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    sonuclar = eller.process(imgRGB)  
#  Görüntüde el olup olmadığını kontrol eder ve el eklemlerini (landmarks) tespit eder.
    print(sonuclar.multi_hand_landmarks)
# Eğer el tespit edilmişse, bu değişkende el eklemlerinin koordinatlarını döner.
# Eklem yerlerine koordinatlarına nokta koyar multi_hand_landmarks

    if sonuclar.multi_hand_landmarks:
        for elLms in sonuclar.multi_hand_landmarks:
            mpCizim.draw_landmarks(img,elLms,mpEl.HAND_CONNECTIONS)
            # draw_landmarks: Elin eklem noktalarını ve bağlantı çizgilerini görüntü üzerine çizer.
            # mpEl.HAND_CONNECTIONS: Eklemler arasındaki bağlantıların çizilmesini sağlar.
            # elLms Eklem yerlerine koordinatlarına nokta koyar multi_hand_landmarks

            for id,lm in enumerate(elLms.landmark): 
		        # enumerate() fonksiyonu, elin tüm eklem noktalarını dönerken hem indeksini (id) hem de o noktaya ait koordinatları (lm) verir.
                h,w,c = img.shape   # (elLms.landmark): MediaPipe'ın tespit ettiği 21 el eklem noktasının listesidir
                cx,cy = int(lm.x*w),int(lm.y*h)
                
                if id ==4:
                    cv2.circle(img,(cx,cy),9,cv2.FILLED)
    
    # fps
    suankiZaman = time.time()
    fps = 1/(suankiZaman-öncekiZaman)
    öncekiZaman=suankiZaman
    
    cv2.putText(img, "FPS : "+str(int(fps)), (5,45), cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0),2)
    cv2.imshow("img",img)
    cv2.waitKey(1)


    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()



