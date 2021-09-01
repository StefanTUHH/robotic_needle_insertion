import numpy as np
import slicer
try:
    from numba import jit, prange
except ImportError:
    slicer.util.pip_install("numba")
    from numba import jit, prange

kernel_size = 2
number_of_kernels = (kernel_size*2+1)**3
kernel = np.zeros((number_of_kernels, 3), dtype=np.int)
counter = 0
for x in range(-kernel_size,kernel_size+1):
    for y in range(-kernel_size,kernel_size+1):
        for z in range(-kernel_size,kernel_size+1):
            kernel[counter, :] = np.asarray((x, y, z))
            counter = counter + 1


def calcDensity(targetPoint: np.ndarray, arrShape: np.ndarray, point_Ijk: np.ndarray, idx: int, imageThreshold: float,
                globalMaxDensity: float, inputVolumeNPArray: np.ndarray, overlayTypeIndex: int, spacing: np.ndarray):
    # if logic.result[point_Ijk[0], point_Ijk[1], point_Ijk[2]] == logic.globalMaxDensity:
    points, remainingPoints = getLine(targetPoint, point_Ijk, arrShape, np.asarray((0, 0, 0)))
    if np.any(inputVolumeNPArray[remainingPoints[:, 0], remainingPoints[:, 1], remainingPoints[:, 2]] > imageThreshold):
        # logic.result[point_Ijk[0], point_Ijk[1], point_Ijk[2]] = logic.globalMaxDensity

        return globalMaxDensity
    if overlayTypeIndex != 4 and overlayTypeIndex != 5:
        densities = inputVolumeNPArray[points[:, 0], points[:, 1], points[:, 2]]
    if overlayTypeIndex == 0:  # max
        maxDensity = np.max(densities)
    elif overlayTypeIndex == 1:  # min
        maxDensity = np.quantile(densities, 0.05)
    elif overlayTypeIndex == 2:  # mean
        maxDensity = np.mean(densities)
    elif overlayTypeIndex == 3:  # standard deviation
        maxDensity = np.std(densities)
    elif overlayTypeIndex == 4:  # distance
        diff = point_Ijk - targetPoint
        diff[0] *= spacing[2]
        diff[1] *= spacing[1]
        diff[2] *= spacing[0]
        maxDensity = np.linalg.norm(diff)
    elif overlayTypeIndex == 5:  # insertion angle
        out_direction = point_Ijk - targetPoint
        out_direction = out_direction / np.linalg.norm(out_direction)
        points_to_check = point_Ijk + kernel
        points_to_check = points_to_check[np.all(points_to_check >= 0, axis=1), :]
        points_to_check = points_to_check[np.all(points_to_check < arrShape, axis=1), :]
        mask = inputVolumeNPArray[points_to_check[:, 0], points_to_check[:, 1], points_to_check[:, 2]] > imageThreshold
        selected_points = np.transpose(points_to_check[mask, :])
        if len(selected_points) == 0:
            maxDensity = globalMaxDensity
        else:
            svd = np.linalg.svd(selected_points - np.mean(selected_points, axis=1, keepdims=True))
            left = svd[0]
            normal = left[:, -1]
            normal = normal / np.linalg.norm(normal)
            angle = np.arccos(np.dot(normal, out_direction))
            if angle > np.pi / 2:
                angle = np.pi - angle
            maxDensity = angle
    else:
        maxDensity = 0
    return maxDensity


def getLine(fromPoint, throughPoint, maxPoint, minPoint, inc=1, asType=np.int):
    gradient = np.asarray(throughPoint - fromPoint, dtype=np.float)
    if np.linalg.norm(gradient) <= 1.:
        points = np.zeros((1, 3))
        points[0, :] = np.round(throughPoint)
        return points, np.asarray(())
    else:
        gradLength = np.linalg.norm(gradient)
        gradient = gradient / gradLength
        diffMax = maxPoint - fromPoint
        diffMin = minPoint - fromPoint
        maxDirection = np.sign(diffMax) == np.sign(gradient)
        iterations = diffMax / gradient * np.asarray(maxDirection, dtype=np.float32) + diffMin / gradient * np.asarray(
            ~maxDirection, dtype=np.float32)
        iterations = int(np.nanmin(iterations))
        throughPointIterations = int(gradLength) + min(5, iterations - int(gradLength))  # Draw the line a bit further
        remainingIterations = iterations - throughPointIterations
        scaledGradients = np.repeat(gradient, throughPointIterations).reshape(
            (3, throughPointIterations)).transpose() * np.arange(throughPointIterations).reshape(throughPointIterations,
                                                                                                 1)
        remainingScaledGradients = np.repeat(gradient, remainingIterations).reshape(
            (3, remainingIterations)).transpose() * np.arange(throughPointIterations, iterations).reshape(
            remainingIterations, 1)

        return np.asarray(scaledGradients + fromPoint, dtype=asType), np.asarray(remainingScaledGradients + fromPoint,
                                                                                 dtype=asType)
