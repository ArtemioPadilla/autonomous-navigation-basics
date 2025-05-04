from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import matplotlib.pyplot as plt
from collections import deque

# Enhanced Hybrid Position+Angle PID Controller
class SteeringController:
    def __init__(self, max_steering_angle=0.5, dead_zone=0.05, 
                 Kp_angle=0.12, Ki_angle=0.0005, Kd_angle=0.07,
                 Kp_position=0.1, Ki_position=0.005, Kd_position=0.01, 
                 smoothing=0.12, position_weight=0.5, monitor=None):
        # Steering limits
        self.max_steering = max_steering_angle
        self.dead_zone = dead_zone
        
        # Angle PID parameters
        self.Kp_angle = Kp_angle
        self.Ki_angle = Ki_angle
        self.Kd_angle = Kd_angle
        
        # Position PID parameters - significantly increased
        self.Kp_position = Kp_position    # Much higher position P gain
        self.Ki_position = Ki_position    # Much higher position I gain
        self.Kd_position = Kd_position    # Higher position D gain
        
        # Smoothing factor
        self.smoothing = smoothing
        
        # Balance between position and angle control (0-1)
        self.position_weight = position_weight
        
        # State variables
        self.prev_steering = 0.0
        self.prev_angle_error = 0.0
        self.prev_position_error = 0.0
        self.integral_angle = 0.0
        self.integral_position = 0.0
        self.max_integral = 0.3
        
        # Overshoot prevention
        self.straight_count = 0
        self.straight_threshold = 8
        self.prev_error_sign = 0
        
        # Monitor
        self.monitor = monitor

    def compute(self, raw_angle, position_offset):
        # IMPORTANT: Position offset is how far the line is from the center
        # Positive offset means line is to the RIGHT of center
        # Negative offset means line is to the LEFT of center
        
        # For steering:
        # Positive steering angle means turning RIGHT
        # Negative steering angle means turning LEFT
        
        # Therefore, to center on the line:
        # If line is to the RIGHT (positive offset), we need to steer RIGHT (positive steering)
        # If line is to the LEFT (negative offset), we need to steer LEFT (negative steering)
        
        # Normalize position offset to be in a similar range as angle (-1 to 1)
        normalized_position = position_offset / 128.0
        
        # Apply dead zone to angle
        if abs(raw_angle) < self.dead_zone:
            angle_for_control = 0.0
            self.straight_count += 1
        else:
            angle_for_control = raw_angle
            self.straight_count = 0
        
        # Detect sign change for overshoot prevention
        current_error_sign = 1 if angle_for_control > 0 else (-1 if angle_for_control < 0 else 0)
        error_sign_changed = (self.prev_error_sign != 0 and current_error_sign != 0 and 
                              self.prev_error_sign != current_error_sign)
        
        if error_sign_changed:
            self.integral_angle *= 0.3  # Reduce integral on sign change
        
        self.prev_error_sign = current_error_sign
            
        # Reset integral when going straight
        if self.straight_count > self.straight_threshold:
            self.integral_angle *= 0.7
            self.integral_position *= 0.7

        # ===== ANGLE PID CONTROL =====
        # Proporcional
        angle_p = self.Kp_angle * angle_for_control
        
        # Integral with anti-windup
        if abs(angle_for_control) > self.dead_zone:
            self.integral_angle = self.integral_angle * 0.95 + angle_for_control
        else:
            self.integral_angle *= 0.85
            
        self.integral_angle = np.clip(self.integral_angle, -self.max_integral, self.max_integral)
        angle_i = self.Ki_angle * self.integral_angle
        
        # Derivative
        angle_d = self.Kd_angle * (angle_for_control - self.prev_angle_error)
        self.prev_angle_error = angle_for_control
        
        # Total angle component
        angle_control = angle_p + angle_i + angle_d
        
        # ===== POSITION PID CONTROL =====
        # For position control: We want to steer TOWARD the line
        # So if position_offset is positive (line to the right), we steer right (positive)
        # This means we use the raw position value directly (no negation needed)
        
        # Proportional - directly use normalized position (no negation)
        position_p = self.Kp_position * normalized_position
        
        # Integral - only accumulate when position error is significant
        if abs(normalized_position) > 0.05:
            self.integral_position = self.integral_position * 0.95 + normalized_position
        else:
            self.integral_position *= 0.85
            
        self.integral_position = np.clip(self.integral_position, -self.max_integral, self.max_integral)
        position_i = self.Ki_position * self.integral_position
        
        # Derivative
        position_d = self.Kd_position * (normalized_position - self.prev_position_error)
        self.prev_position_error = normalized_position
        
        # Total position component
        position_control = position_p + position_i + position_d
        
        # Combine angle and position control using weight
        target_steering = (1 - self.position_weight) * angle_control + self.position_weight * position_control
        
        # Additional overshoot prevention
        if abs(angle_for_control) < 0.1 and abs(normalized_position) < 0.1:
            target_steering *= 0.7  # Reduce steering when close to centered
        
        # Limit steering angle
        target_steering = np.clip(target_steering, -self.max_steering, self.max_steering)
        
        # Smooth transitions
        steering = (1 - self.smoothing) * self.prev_steering + self.smoothing * target_steering
        self.prev_steering = steering
        
        # Send all PID component data to monitor
        if self.monitor:
            self.monitor.update(
                angle_error=angle_for_control, 
                angle_p=angle_p, 
                angle_i=angle_i, 
                angle_d=angle_d,
                position_error=normalized_position, 
                position_p=position_p, 
                position_i=position_i, 
                position_d=position_d, 
                position_output=position_control,
                steering_angle=steering
            )
        
        return steering

# Add a class for real-time PID monitoring
class PIDMonitor:
    def __init__(self, max_points=100):
        self.max_points = max_points
        # Angle PID data
        self.angle_errors = deque(maxlen=max_points)
        self.angle_p = deque(maxlen=max_points)
        self.angle_i = deque(maxlen=max_points)
        self.angle_d = deque(maxlen=max_points)
        self.angle_output = deque(maxlen=max_points)
        
        # Position PID data
        self.position_errors = deque(maxlen=max_points)
        self.position_p = deque(maxlen=max_points)
        self.position_i = deque(maxlen=max_points)
        self.position_d = deque(maxlen=max_points)
        self.position_output = deque(maxlen=max_points)
        
        # Final outputs
        self.steering_angles = deque(maxlen=max_points)

        # Create a separate window for the monitor
        plt.ion()
        
        # Create a more compact figure with 3 subplots
        self.fig = plt.figure(figsize=(12, 9))
        
        # 1. Angle PID components
        # 2. Position PID components
        # 3. Final steering angle
        self.ax1 = self.fig.add_subplot(3, 1, 1)  # Angle components
        self.ax2 = self.fig.add_subplot(3, 1, 2)  # Position components
        self.ax3 = self.fig.add_subplot(3, 1, 3)  # Steering output
        
        # Create all plot lines for angle components
        self.angle_error_line, = self.ax1.plot([], [], label="Error", color="red", linewidth=2)
        self.angle_p_line, = self.ax1.plot([], [], label="P", color="blue", linewidth=1.5)
        self.angle_i_line, = self.ax1.plot([], [], label="I", color="green", linewidth=1.5)
        self.angle_d_line, = self.ax1.plot([], [], label="D", color="orange", linewidth=1.5)
        self.angle_output_line, = self.ax1.plot([], [], label="Output", color="purple", linewidth=2, linestyle='--')
        
        # Position lines - now with full PID components
        self.position_error_line, = self.ax2.plot([], [], label="Error", color="red", linewidth=2)
        self.position_p_line, = self.ax2.plot([], [], label="P", color="blue", linewidth=1.5)
        self.position_i_line, = self.ax2.plot([], [], label="I", color="green", linewidth=1.5)
        self.position_d_line, = self.ax2.plot([], [], label="D", color="orange", linewidth=1.5)
        self.position_output_line, = self.ax2.plot([], [], label="Output", color="purple", linewidth=2, linestyle='--')
        
        # Steering output
        self.steering_line, = self.ax3.plot([], [], label="Steering", color="purple", linewidth=2)
        
        # Configure plots
        self.ax1.set_title("Angle PID Components", fontsize=12, fontweight='bold')
        self.ax2.set_title("Position PID Components", fontsize=12, fontweight='bold')
        self.ax3.set_title("Final Steering Output", fontsize=12, fontweight='bold')
        
        # Add legends with better positioning
        self.ax1.legend(loc='upper right', ncol=5, fontsize=9)
        self.ax2.legend(loc='upper right', ncol=5, fontsize=9)
        self.ax3.legend(loc='upper right', fontsize=10)
        
        # Configure axes
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_xlim(0, max_points)
            ax.set_ylim(-1, 1)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        # Configure specific y limits
        self.ax3.set_ylim(-0.5, 0.5)  # Steering limit
        
        # Add y-axis labels
        self.ax1.set_ylabel("Angle Control", fontweight='bold')
        self.ax2.set_ylabel("Position Control", fontweight='bold')
        self.ax3.set_ylabel("Steering (rad)", fontweight='bold')
        self.ax3.set_xlabel("Time Steps", fontweight='bold')
        
        # Set main title
        self.fig.suptitle("Hybrid Position+Angle PID Controller Monitor", fontsize=14, fontweight='bold')
        
        # Adjust layout
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.92, hspace=0.35)
        
        # Position window
        try:
            mngr = plt.get_current_fig_manager()
            mngr.window.wm_geometry("+900+50")
        except:
            pass

    def update(self, angle_error, angle_p, angle_i, angle_d, 
              position_error, position_p, position_i, position_d, position_output, steering_angle):
        # Store angle PID data
        self.angle_errors.append(angle_error)
        self.angle_p.append(angle_p)
        self.angle_i.append(angle_i)
        self.angle_d.append(angle_d)
        self.angle_output.append(angle_p + angle_i + angle_d)
        
        # Store position PID data
        self.position_errors.append(position_error)
        self.position_p.append(position_p)
        self.position_i.append(position_i)
        self.position_d.append(position_d)
        self.position_output.append(position_output)
        
        # Store final steering output
        self.steering_angles.append(steering_angle)

        # Get x-axis range
        x_range = range(len(self.angle_errors))
        
        # Update all lines - Angle components
        self.angle_error_line.set_data(x_range, self.angle_errors)
        self.angle_p_line.set_data(x_range, self.angle_p)
        self.angle_i_line.set_data(x_range, self.angle_i)
        self.angle_d_line.set_data(x_range, self.angle_d)
        self.angle_output_line.set_data(x_range, self.angle_output)
        
        # Update position components
        self.position_error_line.set_data(x_range, self.position_errors)
        self.position_p_line.set_data(x_range, self.position_p)
        self.position_i_line.set_data(x_range, self.position_i)
        self.position_d_line.set_data(x_range, self.position_d)
        self.position_output_line.set_data(x_range, self.position_output)
        
        # Update steering output
        self.steering_line.set_data(x_range, self.steering_angles)

        # Rescale axes as needed, only the y axis
        for ax in [self.ax1, self.ax2]:
            ax.relim()
            ax.autoscale_view(scaley=True, scalex=False)
            
            # Keep some reasonable bounds on the y-axis
            ymin, ymax = ax.get_ylim()
            if ymax - ymin < 0.1:  # If range is too small
                center = (ymax + ymin) / 2
                ax.set_ylim(center - 0.05, center + 0.05)
            if ymax > 1.0:  # If upper bound is too high
                ax.set_ylim(ymin, 1.0)
            if ymin < -1.0:  # If lower bound is too low
                ax.set_ylim(-1.0, ymax)

        # For steering, keep fixed y range
        self.ax3.relim()
        self.ax3.autoscale_view(scalex=False, scaley=False)
        
        # Enforce proper x-axis limits for all axes
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_xlim(0, len(x_range))

        plt.pause(0.001)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image[:, :, :3]  # Drop alpha

# Image processing config
rho = 1
theta = np.pi / 180
threshold = 15
min_line_len = 30
max_line_gap = 40
alpha = 1
beta = 1
gamma = 1
last_angle = 0.0  # Ensure the initial angle is zero to prevent immediate turning
position_weight = 0.7  # Global variable to share controller weight with display

# ROI mask
ROI_MASK = np.zeros((128, 256), dtype=np.uint8)
vertices = np.array([[(0,128),(0, 100), (100, 80), (156,80), (256,100), (256,128)]], dtype=np.int32)
cv2.fillPoly(ROI_MASK, vertices, 255)

# Image processing
def process_image(image, final_steering=None, raw_angle=None):
    global last_angle

    # Create a copy of the original image for visualization
    img_display = image.copy()
    
    # Visualize ROI with transparency
    roi_display = np.zeros_like(image, dtype=np.uint8)
    # Fill the ROI polygon with a semi-transparent blue color
    cv2.fillPoly(roi_display, [vertices], (100, 50, 0))  # BGR format, reddish-brown
    # Blend ROI visualization with original image
    img_display = cv2.addWeighted(img_display, 1, roi_display, 0.3, 0)

    # Convertir imagen a HSV y filtrar color amarillo
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Adjusted yellow detection range for better detection
    lower_yellow = np.array([15, 70, 70])
    upper_yellow = np.array([45, 255, 255])
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    yellow_only = cv2.bitwise_and(image, image, mask=mask_yellow)

    # Preprocesamiento: gris -> blur -> Canny -> ROI
    gray_img = cv2.cvtColor(yellow_only, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    img_canny = cv2.Canny(img_blur, 50, 350)
    img_mask = cv2.bitwise_and(img_canny, ROI_MASK)

    # Detección de líneas con Hough
    lines = cv2.HoughLinesP(
        img_mask, rho, theta, threshold, np.array([]),
        minLineLength=min_line_len, maxLineGap=max_line_gap
    )

    img_lines = np.zeros((img_mask.shape[0], img_mask.shape[1], 3), dtype=np.uint8)
    best_angle = last_angle
    
    # Get center of image for reference
    h, w = img_lines.shape[:2]
    center_x = w // 2
    
    # Position offset from center
    lane_position_offset = 0

    if lines is not None:
        longest_length = 0
        right_lines = []
        left_lines = []
        all_lines = []
        
        # Separate lines based on their position and slope
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_lines, (x1, y1), (x2, y2), [255, 0, 0], 3)  # líneas en azul
                all_lines.append((x1, y1, x2, y2))
                
                # Determine if line is left or right of center
                mid_x = (x1 + x2) / 2
                if mid_x < center_x:
                    left_lines.append((x1, y1, x2, y2))
                else:
                    right_lines.append((x1, y1, x2, y2))

                # Calcular longitud y ángulo de la línea
                length = np.hypot(x2 - x1, y2 - y1)
                angle_rad = np.arctan2(y2 - y1, x2 - x1)

                # Seleccionar la línea más larga
                if length > longest_length:
                    longest_length = length
                    best_angle = angle_rad
                    # Calculate position offset based on the longest line
                    lane_position_offset = ((x1 + x2) / 2) - center_x

        # Sort lines by y position
        if all_lines:
            all_lines.sort(key=lambda line: min(line[1], line[3]))
            
            # Get top and bottom points of the detected line
            x1_start, y1_start, _, _ = all_lines[0]
            _, _, x2_end, y2_end = all_lines[-1]
            
            cv2.line(img_lines, (x1_start, y1_start), (x2_end, y2_end), (0, 255, 0), 3)  # línea conectada en verde
            
            # Calculate angle between connected line and vertical
            dx = x2_end - x1_start
            dy = y2_end - y1_start
            connected_angle = np.arctan2(dy, dx)
            
            # Use the connected line's angle if it's significant
            if longest_length > 50:
                best_angle = connected_angle

        last_angle = best_angle
    else:
        best_angle = 0.0  # Default to straight if no lines are detected
    # Calculate the angle for steering with adjustments based on lane position
    # Positive = turn right, Negative = turn left
    position_correction = lane_position_offset * 0.001  # Small correction based on position
    steering_angle = -best_angle + position_correction

    # Añadir vector direccional del vehículo - now using final_steering when available
    center_x, center_y = w // 2, h - 20  # Punto de origen del vector (parte inferior central)
    
    # Use the actual final steering angle if provided (this comes from the PID controller output)
    display_angle = final_steering if final_steering is not None else steering_angle
    
    # Draw a simpler and clearer representation of the steering angle
    
    # Scale the vector based on the magnitude of the steering angle
    vector_length = 30  # Fixed vector length for consistent visualization
    angle_radians = display_angle  # Use the steering angle directly
    
    # Calculate the end point of the arrow based on the steering angle
    multiplier = 20  # Adjust this multiplier for better visualization
    end_x = center_x + int(multiplier * vector_length * np.sin(angle_radians))
    end_y = center_y - int(vector_length * np.cos(angle_radians))
    
    # set a maximum length for the arrow
    max_length = 50
    if abs(end_x - center_x) > max_length:
        end_x = center_x + int(max_length * np.sign(end_x - center_x))

    # Convert coordinates to integers for OpenCV
    center_x_int = int(center_x)
    center_y_int = int(center_y)
    end_x_int = int(end_x)
    end_y_int = int(end_y)
    
    # Draw the directional arrow - red for clarity
    cv2.arrowedLine(img_lines, (center_x_int, center_y_int), (end_x_int, end_y_int), (0, 0, 255), 2, tipLength=0.3)  # Thinner line
    # Circle at origin
    cv2.circle(img_lines, (center_x_int, center_y_int), 3, (0, 0, 255), -1)  # Smaller circle
    
    # Añadir marcador de ángulo con arco y valor numérico
    angle_degrees = np.degrees(display_angle)
    
    # Dibujamos un arco que representa el ángulo
    radius = 25
    start_angle = 90  # Comenzamos desde la vertical (90 grados)
    end_angle = 90 - angle_degrees  # El ángulo final
    
    # Determinar el color del arco según la dirección (verde para derecha, azul para izquierda)
    arc_color = (0, 255, 0) if angle_degrees > 0 else (255, 165, 0)
    
    # Dibujamos el arco
    cv2.ellipse(img_lines, (center_x, center_y), (radius, radius), 
                0, start_angle, end_angle, arc_color, 2)
    
    if raw_angle:
        raw_angle_degrees = np.degrees(raw_angle)
        # Replace degree symbol with "deg" text to avoid encoding issues - smaller version
        angle_text = f"{raw_angle_degrees:.1f} deg (raw)"
        text_position = (center_x + 30, center_y - 10)
        cv2.putText(img_lines, angle_text, text_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Replace degree symbol with "deg" text to avoid encoding issues - smaller version
    angle_text = f"{angle_degrees:.1f} deg"
    text_position = (center_x-10, center_y + 10)
    cv2.putText(img_lines, angle_text, text_position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Add position offset indicator - smaller and more compact
    offset_text = f"Off: {lane_position_offset:.1f}px"
    offset_position = (center_x - 80, center_y - 10)
    cv2.putText(img_lines, offset_text, offset_position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # Add indicator to show final vs raw steering when applicable - smaller version
    if final_steering is not None:
        cv2.putText(img_lines, "FINAL", (center_x - 20, center_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
    
    # Smaller radius for the angle arc
    radius = 20
    start_angle = 90  # Comenzamos desde la vertical (90 grados)
    end_angle = 90 - angle_degrees  # El ángulo final
    
    # Determinar el color del arco según la dirección (verde para derecha, azul para izquierda)
    arc_color = (0, 255, 0) if angle_degrees > 0 else (255, 165, 0)
    
    # Dibujamos el arco - thinner line
    cv2.ellipse(img_lines, (center_x, center_y), (radius, radius), 
                0, start_angle, end_angle, arc_color, 1)
    
    # Add controller position/angle influence display - smaller more compact version
    if 'position_weight' in globals():
        pos_pct = int(position_weight * 100)
        ang_pct = 100 - pos_pct
        
        # Create a more compact format for controller mix info
        cv2.putText(img_lines, f"Controller Mix:", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(img_lines, f"Position: {pos_pct}%", (5, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1, cv2.LINE_AA)  
        cv2.putText(img_lines, f"Angle: {ang_pct}%", (5, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)

    # Combine the original image (with transparent ROI) with the lines visualization
    img_lane_lines = cv2.addWeighted(img_display, alpha, img_lines, beta, gamma)
    
    # Retornamos la imagen procesada y el ángulo para el controlador PID
    return img_lane_lines, steering_angle, lane_position_offset

# Display
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
    # Declare global variable at the beginning of the function
    global position_weight
    
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    display_img = Display("display_image")

    keyboard = Keyboard()
    keyboard.enable(timestep)

    speed = 80  # Moderate speed for testing
    pid_monitor = PIDMonitor()
    
    # Initialize the hybrid controller with position-heavy weights
    # These parameters are significantly higher for position controller
    controller = SteeringController(
        max_steering_angle=0.75,
        Kp_angle=0.12, Ki_angle=0.0005, Kd_angle=0.06,
        Kp_position=0.3, Ki_position=0.01, Kd_position=0.02,
        position_weight=0.7,  # 70% position, 30% angle - stronger position influence
        smoothing=0.15,
        monitor=pid_monitor
    )

    # Update global position_weight to match controller's initial value
    position_weight = controller.position_weight

    print("Hybrid Position+Angle Controller active")
    print("Position weight: 70%, Angle weight: 30%")
    print("Press 'P' to increase position influence")
    print("Press 'A' to increase angle influence")

    while robot.step() != -1:
        image = get_image(camera)
        
        # First call process_image to get the raw angle and position_offset
        processed_img, raw_angle, position_offset = process_image(image)
        
        # Compute the final steering angle using the controller
        steering_angle = controller.compute(raw_angle, position_offset)
        
        # Process the image again with the final steering angle for display
        processed_img_with_steering, _, _ = process_image(image, final_steering=steering_angle, raw_angle=raw_angle)
        
        # Display the image with the final steering vector
        display_image(display_img, processed_img_with_steering)

        key = keyboard.getKey()
        if key == ord('S'):
            current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            file_name = current_datetime + ".png"
            print("Image taken")
            camera.saveImage(os.getcwd() + "/" + file_name, 1)
            processed_img = cv2.cvtColor(processed_img_with_steering, cv2.COLOR_BGR2RGB)
        # Keyboard controls for adjusting the position vs angle balance
        elif key == ord('P'):
            controller.position_weight = min(1.0, controller.position_weight + 0.1)
            # Update global variable to reflect in display
            position_weight = controller.position_weight
            print(f"Position influence increased: {controller.position_weight*100:.0f}% position, {(1-controller.position_weight)*100:.0f}% angle")
        elif key == ord('A'):
            controller.position_weight = max(0.0, controller.position_weight - 0.1)
            # Update global variable to reflect in display
            position_weight = controller.position_weight
            print(f"Angle influence increased: {controller.position_weight*100:.0f}% position, {(1-controller.position_weight)*100:.0f}% angle")

        # Apply the final steering angle to the vehicle
        driver.setSteeringAngle(steering_angle)
        driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()