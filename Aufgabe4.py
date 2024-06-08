import numpy as np
from PIL import Image

pool_size = 2
stride = 2

# Bild mit Pillow als schwarz weiß einlesen
input_matrix = np.asarray(Image.open("eevee.png").convert("L"))
input_height, input_width = input_matrix.shape

# kleinere Matrix erzeugen in Abhängigkeit von der Pool Size und Stride
output_height = (input_height - pool_size) // stride + 1
output_width = (input_width - pool_size) // stride + 1
output_matrix = np.zeros((output_height, output_width))

# erzeuge 2x2 pool_regions und trage diese in output_matrix an der entsprechenden Stelle mit Max-Pool ein
for y in range(output_height):
    for x in range(output_width):
        y_1, x_1 = y * stride, x * stride
        y_2, x_2  = y_1 + pool_size, x_1 + pool_size
        pooling_region = input_matrix[y_1:y_2, x_1:x_2]
        output_matrix[y, x] = np.max(pooling_region)


original = Image.fromarray(input_matrix)
new = Image.fromarray(output_matrix)
original.show()
new.show()