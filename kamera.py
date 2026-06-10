import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

fiyatlar = {
    "pepsi": 25.0,
    "7up": 25.0,
    "yedigun": 22.0
}


kamera_id = 0
cap = cv2.VideoCapture(kamera_id, cv2.CAP_DSHOW)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print(f"HATA: {kamera_id} numaralı kamera açılamadı")
    exit()

kare_sayaci = 0
atlama_miktari = 2
son_urun_metni = "Urunler: Bekleniyor..."
son_tutar_metni = "Toplam: 0.0 TL"

print("Kamera açıldı, işlemi durdurmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Kameradan görüntü alınamıyor!")
        break

    kare_sayaci += 1


    if kare_sayaci % atlama_miktari == 0:

        results = model(frame, imgsz=480, conf=0.25, iou=0.6, verbose=False)

        anlik_tutar = 0.0
        ekrandaki_urunler = []

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            sinif_ismi = model.names[class_id]

            if sinif_ismi in fiyatlar:
                anlik_tutar += fiyatlar[sinif_ismi]
                ekrandaki_urunler.append(sinif_ismi)

        son_urun_metni = f"Urunler: {', '.join(ekrandaki_urunler)}"
        son_tutar_metni = f"Toplam: {anlik_tutar} TL"


    cv2.putText(frame, son_urun_metni, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, son_tutar_metni, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Kola Tespiti - Canli Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()