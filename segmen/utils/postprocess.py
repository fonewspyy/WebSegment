import numpy as np

selected_class_indice = [0, 1, 2, 3, 4, 5, 6, 7]
selected_class_rgb = [[0, 0, 0],       # Class 0 (Background) - Black
                      [255, 0, 0],     # Class 1 - Red
                      [0, 255, 0],     # Class 2 - Green
                      [0, 0, 255],     # Class 3 - Blue
                      [255, 255, 0],   # Class 4 - Yellow
                      [0, 255, 255],   # Class 5 - Cyan
                      [255, 165, 0],   # Class 6 - Orange
                      [139, 69, 19]]   # Class 13 - Saddle Brown


def colour_code_segmentation(image):
    colour_code = np.array(selected_class_rgb)
    x = colour_code[image.astype(int)]
    return x