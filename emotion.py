import cv2
from fer import FER

# Kamera başlat
cap = cv2.VideoCapture(0)

# FER modeli oluştur
detector = FER(mtcnn=True)  # MTCNN kullanarak daha doğru yüz algılama

# Duygu simgeleri
emotion_icons = {
    "happy": cv2.imread(r"C:\Users\User\Desktop\happy.png"),
    "sad": cv2.imread(r"C:\Users\User\Desktop\sad.png"),
    "angry": cv2.imread(r"C:\Users\User\Desktop\angry.png"),
    "surprise": cv2.imread(r"C:\Users\User\Desktop\surprise.png"),
    "neutral": cv2.imread(r"C:\Users\User\Desktop\neutral.png"),
    "fear": cv2.imread(r"C:\Users\User\Desktop\fear.png"),
}

def overlay_icon_fixed_position(frame, icon, position):
    """
    Simgeyi sabit bir konuma ekler.
    """
    try:
        x, y = position
        h, w, _ = icon.shape

        # Simgenin ekleneceği alanı seç
        roi = frame[y:y+h, x:x+w]

        # Simgeyi alfa kanalı olmadan eklemek için karıştır
        blended = cv2.addWeighted(roi, 0.5, icon, 0.5, 0)
        frame[y:y+h, x:x+w] = blended
    except Exception as e:
        print(f"Something went wrong: {e}")

# Ekran görüntüsü sayacını başlat
screenshot_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Duygu tespiti
    results = detector.detect_emotions(frame)

    # Her algılanan yüz için
    for result in results:
        (x, y, w, h) = result["box"]
        emotion, score = max(result["emotions"].items(), key=lambda x: x[1])  # En yüksek skorlu duyguyu al

        # Yüze çerçeve çiz
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Duyguyu yüzün üstüne yaz
        cv2.putText(frame, f'{emotion} ({score*100:.1f}%)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Duyguya özgü simgeyi sabit bir konuma yerleştir
        if emotion in emotion_icons:
            icon = emotion_icons[emotion]
            if icon is not None:
                icon_resized = cv2.resize(icon, (100, 100))  # Sabit boyutlandırma
                overlay_icon_fixed_position(frame, icon_resized, (10, 10))  # Simgeyi sol üst köşeye ekle

    # Görüntüyü göster
    cv2.imshow('Face Emotion Recognition', frame)

    # Klavye girdi kontrolü
    key = cv2.waitKey(1) & 0xFF
    if key == ord('x'):  # Çıkış için 'x'
        break
    elif key == ord('s'):  # Ekran görüntüsü için 's'
        screenshot_path = f'screenshot_{screenshot_count}.png'
        cv2.imwrite(screenshot_path, frame)
        print(f"Screenshot saved!: {screenshot_path}")
        screenshot_count += 1

cap.release()
cv2.destroyAllWindows()
