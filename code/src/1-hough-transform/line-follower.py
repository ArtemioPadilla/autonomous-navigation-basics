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
    def __init__(self, max_steering_angle=0.75, dead_zone=0.05, 
                 Kp_angle=0.3, Ki_angle=0.05, Kd_angle=0.07,
                 Kp_position=0.5, Ki_position=0.05, Kd_position=0.2, 
                 smoothing=0.12, position_weight=0.5, monitor=None):
        """
        Initialize the PID controller with given parameters.
        :param max_steering_angle: Maximum steering angle in radians.
        :param
        :param dead_zone: Dead zone for angle control.
        :param Kp_angle: Proportional gain for angle control.
        :param Ki_angle: Integral gain for angle control.
        :param Kd_angle: Derivative gain for angle control.
        :param Kp_position: Proportional gain for position control.
        :param Ki_position: Integral gain for position control.
        :param Kd_position: Derivative gain for position control.
        :param smoothing: Smoothing factor for steering output.
        :param position_weight: Weight for position control (0-1).
        :param monitor: Optional monitor for real-time PID visualization.
        """
        # Steering limits
        self.max_steering = max_steering_angle
        self.dead_zone = dead_zone
        
        # Angle PID parameters
        self.Kp_angle = Kp_angle
        self.Ki_angle = Ki_angle
        self.Kd_angle = Kd_angle
        
        # Position PID parameters
        self.Kp_position = Kp_position
        self.Ki_position = Ki_position
        self.Kd_position = Kd_position
        
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
        # Position offset interpretation:
        # Positive offset = line is to the RIGHT of center
        # Negative offset = line is to the LEFT of center
        #
        # Steering interpretation:
        # Positive steering = turning RIGHT
        # Negative steering = turning LEFT
        
        # Normalize position offset to range (-1 to 1)
        normalized_position = position_offset / 128.0
        
        # Invert raw angle to correct steering direction
        raw_angle = -raw_angle
        
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

        # ANGLE PID CONTROL
        # Proportional
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
        
        # POSITION PID CONTROL
        # For position: Steer toward the line (no negation needed)
        # Positive offset (line to right) = steer right (positive)
        
        # Proportional
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

# Monitor class for real-time PID visualization
class PIDMonitor:
    def __init__(self, max_points=1000):
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

# Get image from camera
def get_image(camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image[:, :, :3]  # Drop alpha channel

# Image processing config
rho = 1  # Distance resolution of the accumulator in pixels
theta = np.pi / 180  # Angle resolution of the accumulator in radians (1 degree)
threshold = 15  # Accumulator threshold parameter. Only lines with enough votes get returned
min_line_len = 30  # Minimum line length. Line segments shorter than this are rejected
max_line_gap = 40  # Maximum allowed gap between points on the same line to link them
alpha = 1  # Weight of the original image when combining with the line image
beta = 1   # Weight of the line image when combining with the original image
gamma = 1  # Added to the sum of the above
last_angle = 0.0  # Initial angle is zero to prevent immediate turning
position_weight = 0.7  # Global variable to share controller weight with display

# ROI mask definition - This restricts line detection to relevant areas of the road
# The polygon defines a trapezoid shape that covers the lower portion of the image where road lines are expected
ROI_MASK = np.zeros((128, 256), dtype=np.uint8)
vertices = np.array([[(0,128),(0, 100), (100, 80), (156,80), (256,100), (256,128)]], dtype=np.int32)
cv2.fillPoly(ROI_MASK, vertices, 255)

# Process image for line detection and visualization
def process_image(image, final_steering=None, raw_angle=None):
    global last_angle

    # Create a copy of the original image for visualization
    img_display = image.copy()
    
    # Initialize raw_line_angle for all code paths
    raw_line_angle = 0.0
    
    # Visualize ROI with transparency
    roi_display = np.zeros_like(image, dtype=np.uint8)
    cv2.fillPoly(roi_display, [vertices], (100, 50, 0))  # Semi-transparent reddish-brown
    img_display = cv2.addWeighted(img_display, 1, roi_display, 0.3, 0)

    # ZEBRA CROSSING DETECTION
    # Get region of interest area for calculations
    roi_area = np.sum(ROI_MASK > 0)
    
    # Convert image to HSV and filter yellow color (to isolate the yellow lane lines and zebra crossings)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 70, 70])
    upper_yellow = np.array([45, 255, 255])
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    
    # Apply ROI mask to yellow mask
    mask_yellow_roi = cv2.bitwise_and(mask_yellow, ROI_MASK)
    
    # Count yellow pixels in the lower part of ROI to detect zebra crossings
    # Create a mask for the bottom third of the ROI where zebras typically appear
    lower_roi_mask = np.zeros_like(ROI_MASK)
    lower_roi_mask[ROI_MASK.shape[0]-40:, :] = ROI_MASK[ROI_MASK.shape[0]-40:, :]
    
    # Apply lower ROI mask to yellow mask
    mask_yellow_lower = cv2.bitwise_and(mask_yellow, lower_roi_mask)
    
    # Count yellow pixels in lower part and calculate density
    yellow_pixel_count_lower = np.sum(mask_yellow_lower > 0)
    yellow_pixel_percentage_lower = (yellow_pixel_count_lower / np.sum(lower_roi_mask > 0)) * 100 if np.sum(lower_roi_mask > 0) > 0 else 0
    
    # Detect zebra crossing based on high density of yellow pixels in lower region
    on_zebra_crossing = yellow_pixel_percentage_lower > 25  # Threshold percentage
    
    # Count total yellow pixels for line detection quality assessment
    yellow_pixel_count = np.sum(mask_yellow_roi > 0)
    yellow_pixel_percentage = (yellow_pixel_count / roi_area) * 100
    
    # Create a copy of the yellow mask for zebra pattern detection
    yellow_mask_for_pattern = mask_yellow.copy()
    
    # If detected probable zebra crossing, analyze the pattern
    if on_zebra_crossing:
        # Erode the yellow mask to separate the stripes
        kernel = np.ones((1, 3), np.uint8)  # Horizontal kernel to separate vertical edges
        eroded_yellow = cv2.erode(yellow_mask_for_pattern, kernel, iterations=1)
        
        # Find contours - each zebra strip should be a separate contour
        contours, _ = cv2.findContours(eroded_yellow & lower_roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count contours in lower part that meet zebra strip criteria
        strip_count = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Check if the shape is rectangular and appropriate size for zebra strip
            aspect_ratio = float(w) / h if h > 0 else 0
            area = w * h
            if area > 20 and aspect_ratio > 1.5:  # Wide and short rectangles
                strip_count += 1
        
        # Confirm zebra crossing based on strip pattern
        on_zebra_crossing = on_zebra_crossing and strip_count > 2
    
    # Prepare yellow pixels for line detection
    yellow_only = cv2.bitwise_and(image, image, mask=mask_yellow)

    # If on a zebra crossing, we need to filter out the zebra stripes
    if on_zebra_crossing:
        # Create a mask that excludes the bottom part where zebra crossings are
        upper_part_mask = np.zeros_like(ROI_MASK)
        upper_part_mask[:ROI_MASK.shape[0]-35, :] = ROI_MASK[:ROI_MASK.shape[0]-35, :]
        
        # Apply the upper part mask to get only upper yellow lines (lane markers, not zebras)
        yellow_only_filtered = cv2.bitwise_and(yellow_only, yellow_only, mask=upper_part_mask)
        
        # Use this filtered image for line detection
        gray_img = cv2.cvtColor(yellow_only_filtered, cv2.COLOR_BGR2GRAY)
    else:
        # Normal processing without zebra filtering
        gray_img = cv2.cvtColor(yellow_only, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    # Apply Canny edge detection to find edges
    img_canny = cv2.Canny(img_blur, 50, 350)
    
    # Apply ROI mask to focus only on relevant road area
    img_mask = cv2.bitwise_and(img_canny, ROI_MASK)

    # Line detection using Hough Transform with parameters that depend on zebra crossing detection
    if on_zebra_crossing:
        # More strict parameters when on zebra crossings
        min_line_length = 40  # Longer minimum line length
        max_line_gap_val = 20  # Smaller gap to avoid connecting zebra strips
        threshold_val = 25     # Higher threshold for more confident lines
    else:
        # Standard parameters for normal road conditions
        min_line_length = min_line_len
        max_line_gap_val = max_line_gap
        threshold_val = threshold
    
    # Get lines with appropriate parameters
    lines = cv2.HoughLinesP(
        img_mask, rho, theta, threshold_val, np.array([]),
        minLineLength=min_line_length, maxLineGap=max_line_gap_val
    )

    img_lines = np.zeros((img_mask.shape[0], img_mask.shape[1], 3), dtype=np.uint8)
    best_angle = last_angle
    
    # Get center of image for reference
    h, w = img_lines.shape[:2]
    center_x = w // 2
    
    # Position offset from center
    lane_position_offset = 0

    # Process the detected lines to determine steering angle
    if lines is not None:
        # Number of lines detected by Hough Transform
        num_lines_detected = len(lines)
        
        longest_length = 0
        right_lines = []
        left_lines = []
        all_lines = []
        
        # Process detected lines
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_lines, (x1, y1), (x2, y2), [255, 0, 0], 3)  # Blue lines
                all_lines.append((x1, y1, x2, y2))
                
                # Determine if line is left or right of center
                mid_x = (x1 + x2) / 2
                if mid_x < center_x:
                    left_lines.append((x1, y1, x2, y2))
                else:
                    right_lines.append((x1, y1, x2, y2))

                # Calculate length and angle of the line
                length = np.hypot(x2 - x1, y2 - y1)
                
                # Calculate angle using arctan2 (angle relative to horizontal axis)
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                
                # Filter out horizontal lines (which can cause erratic steering)
                # Calculate angle from horizontal (0 degrees is horizontal)
                angle_from_horizontal = abs(angle_rad)  # This will be close to 0 for horizontal lines
                if angle_from_horizontal < 0.3 or angle_from_horizontal > (np.pi - 0.3):
                    # Skip nearly horizontal lines (±17 degrees from horizontal)
                    continue
                
                # Convert to vertical-relative angle
                # Subtract π/2 (90 degrees) to make vertical = 0 degrees
                vertical_relative_angle = angle_rad - (np.pi/2)
                
                # Normalize the angle to prevent jumps between -180 and 180 degrees
                vertical_relative_angle = normalize_angle(vertical_relative_angle)
                
                
                # Store the longest line's angle
                if length > longest_length:
                    longest_length = length
                    best_angle = vertical_relative_angle
                    raw_line_angle = vertical_relative_angle
                    lane_position_offset = ((x1 + x2) / 2) - center_x

        # Sort lines by y position (from top to bottom)
        if all_lines:
            all_lines.sort(key=lambda line: min(line[1], line[3]))
            
            # Get top and bottom points of the detected line
            # This creates a single line representing the lane direction
            x1_start, y1_start, _, _ = all_lines[0]
            _, _, x2_end, y2_end = all_lines[-1]
            
            # Draw a green line connecting the top and bottom points
            cv2.line(img_lines, (x1_start, y1_start), (x2_end, y2_end), (0, 255, 0), 3)  # Green line
            
            # Calculate angle between connected line and vertical
            dx = x2_end - x1_start
            dy = y2_end - y1_start
            connected_angle = np.arctan2(dy, dx)
            
            # Convert connected angle to be relative to vertical
            connected_vertical_angle = connected_angle - (np.pi/2)
            
            # Use the connected line's angle if it's significant length
            # This helps to get a more stable line direction
            if longest_length > 50:
                best_angle = connected_vertical_angle
                raw_line_angle = connected_vertical_angle

        # Update the last known angle
        last_angle = best_angle
    else:
        # Default to straight if no lines detected
        # This handles the case of intersections where road has no yellow lines
        best_angle = 0.0  
        raw_line_angle = 0.0
        
    # Calculate steering angle with position adjustment
    # Positive = turn right (clockwise), Negative = turn left (counter-clockwise)
    # The position_correction adds a small adjustment based on lateral position of the lane
    position_correction = lane_position_offset * 0.001  # Small correction based on position
    
    # Steering is opposite to line lean
    steering_angle = -best_angle + position_correction

    # Add vehicle direction vector
    center_x, center_y = w // 2, h - 20  # Origin point at bottom center
    
    # Use actual final steering angle if provided
    display_angle = final_steering if final_steering is not None else steering_angle
    
    # Draw a representation of the steering angle
    vector_length = 30  # Fixed vector length
    angle_radians = display_angle
    
    # Calculate arrow endpoint based on steering angle
    multiplier = 20
    end_x = center_x + int(multiplier * vector_length * np.sin(angle_radians))
    end_y = center_y - int(vector_length * np.cos(angle_radians))
    
    # Set maximum length for the arrow
    max_length = 50
    if abs(end_x - center_x) > max_length:
        end_x = center_x + int(max_length * np.sign(end_x - center_x))

    # Convert coordinates to integers
    center_x_int = int(center_x)
    center_y_int = int(center_y)
    end_x_int = int(end_x)
    end_y_int = int(end_y)
    
    # Draw directional arrow
    cv2.arrowedLine(img_lines, (center_x_int, center_y_int), (end_x_int, end_y_int), (0, 0, 255), 2, tipLength=0.3)
    cv2.circle(img_lines, (center_x_int, center_y_int), 3, (0, 0, 255), -1)
    
    # Add angle marker with arc and numeric value
    angle_degrees = np.degrees(display_angle)
    
    # Draw arc representing the angle
    radius = 25
    start_angle = 90  # Start from vertical
    end_angle = 90 - angle_degrees  # End angle
    
    # Determine arc color by direction (green for right, orange for left)
    arc_color = (0, 255, 0) if angle_degrees > 0 else (255, 165, 0)
    
    # Draw the arc
    cv2.ellipse(img_lines, (center_x, center_y), (radius, radius), 
                0, start_angle, end_angle, arc_color, 2)
    
    if best_angle is not None:
        vertical_relative_angle = best_angle
        vertical_relative_angle_degrees = np.degrees(vertical_relative_angle)
        # Display raw angle
        angle_text = f"{vertical_relative_angle_degrees:.1f} deg (line)"
        text_position = (center_x + 30, center_y - 10)
        cv2.putText(img_lines, angle_text, text_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Display steering angle
    angle_text = f"{angle_degrees:.1f} deg (steering)"
    text_position = (center_x-20, center_y + 10)
    cv2.putText(img_lines, angle_text, text_position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Display position offset
    offset_text = f"Off: {lane_position_offset:.1f}px"
    offset_position = (center_x - 80, center_y - 10)
    cv2.putText(img_lines, offset_text, offset_position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Display zebra crossing indicator
    if on_zebra_crossing:
        zebra_text = "ZEBRA CROSSING"
        cv2.rectangle(img_lines, (center_x - 60, center_y - 45), (center_x + 60, center_y - 25), (0, 0, 0), -1)
        cv2.putText(img_lines, zebra_text, (center_x - 55, center_y - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
        
        # When on a zebra crossing, we should maintain previous steering direction
        # to ensure stability while crossing
        steering_angle = steering_angle * 0.3 + last_angle * 0.7
    
    # Indicate if showing final vs raw steering
    if final_steering is not None:
        cv2.putText(img_lines, "FINAL", (center_x - 20, center_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
    
    # Smaller radius for the angle arc
    radius = 20
    start_angle = 90
    end_angle = 90 - angle_degrees
    
    # Draw the arc with thinner line
    cv2.ellipse(img_lines, (center_x, center_y), (radius, radius), 
                0, start_angle, end_angle, arc_color, 1)
    
    # Display controller mix info
    if 'position_weight' in globals():
        pos_pct = int(position_weight * 100)
        ang_pct = 100 - pos_pct
        
        cv2.putText(img_lines, f"Controller Mix:", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(img_lines, f"Position: {pos_pct}%", (5, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1, cv2.LINE_AA)  
        cv2.putText(img_lines, f"Angle: {ang_pct}%", (5, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)

    # Combine images
    img_lane_lines = cv2.addWeighted(img_display, alpha, img_lines, beta, gamma)
    
    # Return processed image and control values
    return img_lane_lines, steering_angle, lane_position_offset

# Display the image
def display_image(display, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

def normalize_angle(angle):
    """
    Normalize angle to be in the range [-π/2, π/2] to prevent jumps 
    between -180 and 180 degrees.
    """
    # First ensure the angle is in the range [-π, π]
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    
    # Handle the case where angles near -π/2 and π/2 might jump
    if angle > np.pi/2:
        angle = angle - np.pi
    elif angle < -np.pi/2:
        angle = angle + np.pi
        
    return angle

# Main function
def main():
    global position_weight
    
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    display_img = Display("display_image")

    # Initialize keyboard for user input
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # Controller and driving parameters
    CRUISE_SPEED = 80  # Moderate speed for testing
    
    # PID controller constants
    STEERING_MAX = 0.75
    DEAD_ZONE = 0.05
    SMOOTHING = 0.1
    
    # Angle control parameters
    ANGLE_KP = 0.05
    ANGLE_KI = 0.05
    ANGLE_KD = 0.07
    
    # Position control parameters
    POSITION_KP = 0.5
    POSITION_KI = 0.05
    POSITION_KD = 0.2
    
    # Balance between position and angle control (0.7 = 70% position, 30% angle)
    INITIAL_POSITION_WEIGHT = 0.7
    
    # Initialize PID monitor for real-time visualization
    pid_monitor = PIDMonitor()
    
    # Initialize hybrid controller with defined parameters
    controller = SteeringController(
        max_steering_angle=STEERING_MAX, 
        dead_zone=DEAD_ZONE, 
        Kp_angle=ANGLE_KP, 
        Ki_angle=ANGLE_KI, 
        Kd_angle=ANGLE_KD,
        Kp_position=POSITION_KP, 
        Ki_position=POSITION_KI, 
        Kd_position=POSITION_KD, 
        position_weight=INITIAL_POSITION_WEIGHT,
        smoothing=SMOOTHING,
        monitor=pid_monitor
    )

    # Update global position_weight to match controller's initial value
    position_weight = controller.position_weight

    # Display controller information and user instructions
    print("\n== Hybrid Position+Angle Controller ==")
    print(f"Position weight: {int(position_weight*100)}%, Angle weight: {int((1-position_weight)*100)}%")
    print("User controls:")
    print("  'P' - Increase position influence")
    print("  'A' - Increase angle influence")
    print("  'S' - Save screenshot")

    while robot.step() != -1:
        image = get_image(camera)
        
        # Get raw angle and position_offset
        processed_img, raw_angle, position_offset = process_image(image)
        
        # Compute final steering angle using controller
        offset_target = 0
        steering_angle = controller.compute(raw_angle, position_offset - offset_target)
        
        # Process image again with final steering angle for display
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
        # Keyboard controls for adjusting position vs angle balance
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

        # Apply steering angle to vehicle
        driver.setSteeringAngle(steering_angle)
        driver.setCruisingSpeed(CRUISE_SPEED)


if __name__ == "__main__":
    main()