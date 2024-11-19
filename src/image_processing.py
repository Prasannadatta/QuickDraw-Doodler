import cv2
import numpy as np

def full_strokes_to_vector_images(sample):
    """
    Converts a sample from 'full' stroke format to the format expected by vector_to_raster.

    Parameters:
    - sample: A single sample from the dataset, which is a NumPy array of shape (4, N),
      where the rows are x, y, t, p.

    Returns:
    - simplified_image: A list of strokes, where each stroke is a NumPy array of shape (n_points, 2).
    """
    x, y, t, p = sample  # extract the components
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    p = p.astype(np.uint8)

    simplified_image = []
    current_stroke = []

    for idx in range(len(p)):
        if p[idx] == 0 or p[idx] == 1:
            # pen is down or pen start, add point to current stroke
            current_stroke.append([x[idx], y[idx]])
        elif p[idx] == 2 or p[idx] == 3:
            # pen up or pen end, add point to current stroke and save it
            current_stroke.append([x[idx], y[idx]])
            simplified_image.append(np.array(current_stroke, dtype=np.float32))
            current_stroke = []  # Reset current stroke

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
    Takes in stroke data and renders it.
    Additionally centers the doodle in the output image and scales down applying antialiasing in the process.
    
    Parameters:
    - vector_images: List of vector images, where each vector image is a list of strokes,
      and each stroke is a NumPy array of shape (n_points, 2).
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
        # blank canvas with the background color (height, width, channels)
        img = np.full((out_size, out_size, 3), bg_color, dtype=np.uint8)

        # collect all points and calculate bbox
        all_points = np.concatenate([stroke for stroke in vector_image if len(stroke) > 0])
        bbox_min = all_points.min(axis=0)
        bbox_max = all_points.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2.0
        img_center = np.array([out_size / 2.0, out_size / 2.0])

        # scaling factor
        scale = (out_size - total_padding) / max(bbox_max - bbox_min)
        scale *= 0.95  # Slightly reduce scale to fit within the image

        # draw strokes with anti-aliasing
        for stroke in vector_image:
            if len(stroke) == 0:
                continue
            # center and scale the stroke
            transformed_stroke = (stroke - bbox_center) * scale + img_center
            pts = transformed_stroke.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=False, color=fg_color,
                          thickness=thickness, lineType=cv2.LINE_AA)

        # convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raster_images[i] = img_gray

    return raster_images