import random
import math
import sys
import numpy as np

ITERATIONS = 15
INPUT = 'input.txt'


def read_file(input_name):

    with open(input_name) as file:
        points = np.array([])

        threshold = float(next(file))  # read first line
        points_num = int(next(file))  # read second line

        for line in file: # read rest of lines
            point = [float(x) for x in line.split()]
            points = np.append(points, point, axis=0)

        points = points.reshape(-1, 3)
        assert(len(points) == points_num)

        return threshold, points_num, points


def fit_plane(data):
    A = np.c_[data[0], data[1], np.ones(data[0].shape)]
    coeff,_,_,_ = np.linalg.lstsq(A, data[2], rcond=None)
    return -coeff[0], -coeff[1], 1, -coeff[2]


def distance_to_plane(x, y, z, a, b, c, d, sqrtabc):
    return math.fabs(a * x + b * y + c * z + d) / sqrtabc


def ransac(points_num, points, num_of_iterations, threshold):

    best_inliers = np.array([])

    if points_num < 3:
        return 0, 0, 0, 0

    if points_num == 3:
        num_of_iterations = 1

    for i in range(0, num_of_iterations):
        sp1, sp2, sp3 = random.sample(range(0, points_num), 3)
        selected_points = np.array([points[sp1], points[sp2], points[sp3]])
        transp_data = selected_points.transpose()
        a, b, c, d = fit_plane(transp_data)
        inliers = np.array([])

        sqrtabc = math.sqrt(a * a + b * b + c * c) + sys.float_info.epsilon

        for point_index in range(0, points_num):
            current_point = points[point_index]
            distance = distance_to_plane(current_point[0], current_point[1], current_point[2], a, b, c, d, sqrtabc)

            if distance <= threshold:
                inliers = np.append(inliers, current_point)

        inliers_len = len(inliers) / 3

        if (inliers_len > points_num / 2) and inliers_len > len(best_inliers):
            best_inliers = inliers.reshape(-1, 3)

    best_inliers = best_inliers.transpose()
    return fit_plane(best_inliers)


threshold, points_num, points = read_file(INPUT)

print("%.6f %.6f %.6f %.6f" %(ransac(points_num, points, ITERATIONS, threshold)))
