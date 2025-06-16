import cv2
import mediapipe
import time
import random
import pyttsx3
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import threading

# Список русских букв
letters = ["А", "Б", "В", "Г", "Д", "Е", "Ё", "Ж", "З", "И"]

mp_hands = mediapipe.solutions.hands
hands_mesh = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Инициализация озвучивания
engine = pyttsx3.init()

# Установите скорость речи (значение по умолчанию обычно 200)
engine.setProperty('rate', 160)  # Уменьшите скорость чтения немного (например, 175)

cap = cv2.VideoCapture(0)

pocket = []
start_time = time.time()
current_letter = random.choice(letters)

frame_counter = 0
skip_frames = 3  # Пропускать 2 кадра из 3 для ускорения

if not cap.isOpened():
    raise IOError("Ошибка видеозахвата")

# Путь к шрифту для отображения текста
font_path = "C:\\Windows\\Fonts\\arial.ttf"  # Убедитесь, что у вас есть подходящий шрифт
font = ImageFont.truetype(font_path, 300)

# Генерация случайного цвета в формате RGB
def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

marker_color = generate_random_color()

def speak(text):
    engine.say(text)  # Оговорить текст
    engine.runAndWait()  # Дождаться, пока речь будет произнесена

def speak_with_delay(letter, delay):
    time.sleep(delay)  # Задержка перед озвучиванием
    # Формируем фразу для озвучивания
    text_to_speak = f"Напиши букву {letter}"
    speak(text_to_speak)

try:
    # Озвучиваем начальную букву с задержкой
    t = threading.Thread(target=speak_with_delay, args=(current_letter, 2.5))  # Задержка 2.5 секунды
    t.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)

        frame_counter += 1
        if frame_counter % skip_frames != 0:
            continue

        results = hands_mesh.process(frame)

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text((220, 50), current_letter, font=font, fill=(255, 255, 255))

        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                elem = hand.landmark[8]
                x = int(elem.x * frame.shape[1])
                y = int(elem.y * frame.shape[0])
                cv2.circle(frame, (x, y), 20, marker_color, -1)

                for i in pocket:
                    cv2.circle(frame, i, 20, marker_color, -1)
                pocket.append((x, y))

                elapsed_time = time.time() - start_time
                if elapsed_time > 15:
                    pocket = []
                    current_letter = random.choice(letters)
                    marker_color = generate_random_color()

                    # Запускаем озвучивание новой буквы в новом потоке
                    t = threading.Thread(target=speak_with_delay, args=(current_letter, 1))  # Задержка 3 секунды
                    t.start()  # Озвучивание буквы
                    start_time = time.time()

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("d"):
            break

except Exception as e:
    print(f"Ошибка: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows() 

