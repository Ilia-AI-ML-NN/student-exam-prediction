import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

X = np.array([[1,2], [2,1], [3,4], [4, 3], [5,5], [6,4]])
Y = np.array([0, 0, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X, Y)

new_student = np.array([[3, 2]]) 
predicted_class = model.predict(new_student)
probability = model.predict_proba(new_student)

print("прогноз: студент сдаст экзамен" if predicted_class[0] == 1 else "прогноз: студент не сдаст экзамен")
print("Вероятность сдачи экзамена:", probability[0][1])

plt.figure(figsize=(8, 6))

labels_added = set()

for i in range(len(X)):
    color = 'green' if Y[i] == 1 else 'red'
    label = f"Класс {Y[i]}" if Y[i] not in labels_added else None
    plt.scatter(X[i][0], X[i][1], color=color, label=label, s=60)
    labels_added.add(Y[i])
  
plt.scatter(new_student[0][0], new_student[0][1], color = 
            'blue', marker = 'x', s = 100, label = 'новый студент')

x_min, x_max = 0, 7
y_min, y_max = 0, 7
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha = 0.2, cmap = 'RdBu')


plt.xlabel("часы подготовки")
plt.ylabel("решённых заданий")
plt.title("логистическая регрессия")
plt.legend()
plt.grid(True)
plt.savefig("exam_prediction.png", dpi=300)
plt.show()
