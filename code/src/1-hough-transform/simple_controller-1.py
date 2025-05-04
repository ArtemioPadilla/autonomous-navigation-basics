"""camera_pid controller with Hough Transform."""

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os

# ROI mask
ROI_MASK = np.zeros((128, 256), dtype=np.uint8)
vertices = np.array([[(0,128),(0, 110), (90, 80), (166,80), (256,110), (256,128)]], dtype=np.int32)
cv2.fillPoly(ROI_MASK, vertices, 255)

# Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    image = cv2.bitwise_and(image[:, :, :3], image[:, :, :3], mask=ROI_MASK)  # Apply ROI mask
    return image

# Image processing
def process_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect edges using Canny
    edges = cv2.Canny(blur, 50, 150)

    edges = cv2.bitwise_and(edges, edges, mask=ROI_MASK)
    return edges

# Detect lane lines using Hough Transform
def detect_lanes(edges):
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=150
    )
    return lines

# Calculate steering angle based on lane lines
def calculate_steering_angle(lines, image_width):
    if lines is None:
        return 0.0  # No lines detected, go straight

    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
        if slope < -0.5:  # Left lane
            left_lines.append(line)
        elif slope > 0.5:  # Right lane
            right_lines.append(line)

    left_x = np.mean([line[0][0] for line in left_lines]) if left_lines else 0
    right_x = np.mean([line[0][0] for line in right_lines]) if right_lines else image_width

    lane_center = (left_x + right_x) / 2
    image_center = image_width / 2

    # Calculate steering angle (proportional to lane center offset)
    return (image_center - lane_center) / image_width

# Display image
def display_image(display, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

# Main
def main():
    # Create the Robot instance.
    robot = Car()
    driver = Driver()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # Processing display
    display_img = Display("display_image")

    # Create keyboard instance
    keyboard = Keyboard()
    keyboard.enable(timestep)

    global speed
    speed = 15

    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)

        # Process image
        edges = process_image(image)
        lines = detect_lanes(edges)
        steering_angle = calculate_steering_angle(lines, image.shape[1])

        # Display processed image
        display_image(display_img, edges)

        # Update steering angle and speed
        driver.setSteeringAngle(steering_angle)
        driver.setCruisingSpeed(speed)

        # Read keyboard
        key = keyboard.getKey()
        if key == keyboard.UP:  # Increase speed
            speed += 5.0
        elif key == keyboard.DOWN:  # Decrease speed
            speed -= 5.0
        elif key == ord('A'):  # Save image
            current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            file_name = current_datetime + ".png"
            print("Image taken")
            camera.saveImage(os.getcwd() + "/" + file_name, 1)

if __name__ == "__main__":
    main()
