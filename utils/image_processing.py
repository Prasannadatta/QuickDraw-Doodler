import cv2
import numpy as np

def full_strokes_to_vector_images(full_strokes):
    """
    Converts vector images from the 'full' format to the 'simplified' format efficiently using NumPy.

    Parameters:
    - full_strokes: List of vector images in 'full' format, where each vector image is a list of strokes,
      and each stroke is a list containing [x_values], [y_values], [t_values].

    Returns:
    - vector_images_simplified: List of vector images in simplified format, where each vector image is a list of strokes,
      and each stroke is a NumPy array of shape (n_points, 2).
    """
    simplified_image = []
    for stroke in full_strokes:
        # Extract x and y values as NumPy arrays
        x_values = np.array(stroke[0], dtype=np.float32)
        y_values = np.array(stroke[1], dtype=np.float32)
        # Stack x and y into a (n_points, 2) array
        points = np.stack((x_values, y_values), axis=-1)
        simplified_image.append(points)
    return simplified_image

def vector_to_raster(
        vector_images,
        in_size=256,
        out_size=28,
        line_diameter=2,
        padding=2,
        bg_color=(255, 255, 255),
        fg_color=(0, 0, 0)
    ):
    """
    Convert squared vector images to square raster images using OpenCV.
    Takes in stroke data of quickdraw stroke data and renders it.
    Additionally will center the doodle in the output image and scale down applying antialiasing in the process
    
    Parameters:
    - vector_images: List of vector images, where each vector image is a list of strokes,
      and each stroke is a list of (x, y) tuples.
    - data_mode: enum indicating if full stroke data or simplified (different input data format)
    - in_size: The size of the square input image (width and height).
    - out_size: The size of the square output image (width and height).
    - line_diameter: The thickness of the strokes.
    - padding: Padding around the image.
    - bg_color: Background color as an RGB tuple.
    - fg_color: Foreground color (stroke color) as an RGB tuple.

    Returns:
    - raster_images: List of rasterized images as NumPy arrays.
    """

    total_padding = padding * 2.0 + line_diameter
    new_scale = out_size / (in_size + total_padding)
    thickness = max(int(line_diameter * new_scale), 1)

    raster_images = np.zeros((len(vector_images), out_size, out_size), dtype=np.uint8)
    for i, vector_image in enumerate(vector_images):
        # blank canvas with the background color (width, height, depth->rgb)
        img = np.full((out_size, out_size, 3), bg_color, dtype=np.uint8)

        # collect all points and calculate bbox
        all_points = np.concatenate([np.array(stroke, dtype=np.float32) for stroke in vector_image])
        bbox_min = all_points.min(axis=0)
        bbox_max = all_points.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2.0
        img_center = in_size / 2.0

        # offset to center the strokes
        offset = img_center - bbox_center

        # translation and scaling factor of individual strokes
        translation = (offset + total_padding / 2.0) * new_scale

        # draw strokes with anti-aliasing
        for stroke in vector_image:
            stroke_array = np.array(stroke, dtype=np.float32)

            # translate and scale all vectors (strokes)
            transformed_stroke = stroke_array * new_scale + translation
            pts = transformed_stroke.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=False, color=fg_color,
                          thickness=thickness, lineType=cv2.LINE_AA)

        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raster_images[i] = img_gray

    return raster_images