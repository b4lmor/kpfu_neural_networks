import os
import numpy as np
from PIL import Image
from network import Network

def load_single_image(image_path):
    img = Image.open(image_path).convert('L')  # Преобразуем в градации серого
    img = img.resize((28, 28))  # Убедимся, что размер 28x28
    img_array = np.array(img).reshape(28*28, 1)  # Преобразуем в вектор
    return img_array

def load_data(data_dir):
    training_data = []
    for digit in range(0, 4):  # Цифры от 1 до 4
        for index in range(1, 3):  # 4 изображения для каждой цифры
            filename = f"digit_{digit}_{index}.jpg"
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                # Загружаем изображение и преобразуем его в массив
                img = Image.open(filepath).convert('L')  # Преобразуем в градации серого
                img = img.resize((28, 28))  # Убедимся, что размер 28x28
                img_array = np.array(img).reshape(28*28, 1)  # Преобразуем в вектор
                y = np.zeros((4, 1))  # Вектор для меток
                y[digit - 1] = 1  # Устанавливаем 1 для правильного класса
                training_data.append((img_array, y))
    return training_data

data_dir = "samples"  # Укажите путь к папке с изображениями
training_data = load_data(data_dir)

# Создание нейросети
sizes = [28*28, 30, 4]  # Входной слой, скрытый слой, выходной слой
net = Network(sizes)

# Обучение нейросети
epochs = 30
mini_batch_size = 10
eta = 3.0  # Скорость обучения
net.SGD(training_data, epochs, mini_batch_size, eta)

# Пример тестовых данных (можно использовать те же изображения, но с другими индексами)
test_data = load_data(data_dir)  # Или загрузите другой набор данных

# Оценка нейросети
accuracy = net.evaluate(test_data)
print(f"Точность на тестовых данных: {accuracy} из {len(test_data)}")



# Путь к папке с изображениями
data_dir = "samples"

image_path = os.path.join(data_dir, f"digit_0_1.jpg")

# Загрузка изображения
input_image = load_single_image(image_path)

# Получение предсказания от нейросети
output = net.feedforward(input_image)

# Определение предсказанного класса
predicted_class = np.argmax(output)

# Вывод результата
print(f"Предсказанная цифра: {predicted_class}")
