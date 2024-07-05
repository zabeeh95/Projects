import numpy as np

# Define lengths
L2, L3, L4, L5 = 10.5, 9, 2, 15.5
r = 17  # Assuming r is constant

# Define limits (not used in this script but included for completeness)
limit_theta0, limit_theta1, limit_theta2, limit_theta3 = 181, 181, 146, 161

# Initialize theta values directly
theta = [
    np.deg2rad(45),  # theta_1
    np.deg2rad(74),  # theta_2
    np.deg2rad(3),   # theta_3
    np.deg2rad(80)   # theta_4
]

# Define trigonometric functions for convenience
cos, sin, tan = np.cos, np.sin, np.tan

# Calculate forward kinematics
fwd_x = cos(theta[0]) * (L2 * cos(theta[1]) + L3 * cos(theta[1] - theta[2]) + r * cos(theta[1] - theta[2] - theta[3]))
fwd_y = sin(theta[0]) * (L2 * cos(theta[1]) + L3 * cos(theta[1] - theta[2]) + r * cos(theta[1] - theta[2] - theta[3]))
fwd_z = L2 * sin(theta[1]) + L3 * sin(theta[1] - theta[2]) + r * sin(theta[1] - theta[2] - theta[3])

# Print calculated position
print(f"x: {fwd_x:.2f}; y: {fwd_y:.2f}; z: {fwd_z:.2f}")
