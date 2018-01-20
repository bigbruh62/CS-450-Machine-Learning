import numpy as np

matrix = [[1.2, 3.6, 2.8, "sick"],
          [2.6, 4.6, 1.1, "healthy"],
          [3.1, 2.5, 3.4, "sick"],
          [2.2, 2.6, 2.1, "sick"],
          [9.6, 3.8, 5.2, "healthy"],
          [3.3, 1.3, 8.3, "sick"],
          [2.1, 2.3, 2.3, "healthy"],
          [1.2, 3.1, 5.1, "healthy"],
          [1.5, 2.5, 1.6, "healthy"]]

point = [3.7, 2.8, 4.0, ""]

healthy_count = 0
sick_count = 0

distances = []

def euclidean_dist(point1, point2):
    distance = (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2
    return distance

for x in matrix:
    distances.append(euclidean_dist(point, x))

print(distances)

print(np.argsort(distances))

k = 3

closest = np.argsort(distances)[0:k]

print(closest)

for i in closest:
    if matrix[i][3] == "sick":
        sick_count += 1
    else:
        healthy_count += 1

if healthy_count > sick_count:
    point[3] = "healthy"
else:
    point[3] = "sick"

print(point)