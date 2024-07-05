import numpy as np
import sys

# Define lengths and limits
L2, L3, L4, L5 = 10.5, 9, 2, 15.5
r = 17  # Assuming r is constant
limit_theta = [180, 180, 150, 160]

theta = [0, 0, 0, 0]

# Define trigonometric functions for convenience
cos, sin, tan = np.cos, np.sin, np.tan

# Set Cartesian coordinates directly
x, y, z = 15, 15, 15

# Calculate r1
r1 = np.sqrt(x ** 2 + y ** 2)

# Check for self-collision or below surface
if r1 < 7.2 and z < 10:
    print("Self Collision !!!")
    sys.exit()
if z < -5:
    print("End effector is trying to move below surface !!!")
    sys.exit()

# Surrounding workspace restriction check
if r1 < 8 or r1 > 28:
    print("Spherical Workspace Restriction !!!")
    sys.exit()

def calculate_forward_kinematics(theta):
    """Calculate forward kinematics."""
    fwd_x = cos(theta[0]) * (L2 * cos(theta[1]) + L3 * cos(theta[1] - theta[2]) + r * cos(theta[1] - theta[2] - theta[3]))
    fwd_y = sin(theta[0]) * (L2 * cos(theta[1]) + L3 * cos(theta[1] - theta[2]) + r * cos(theta[1] - theta[2] - theta[3]))
    fwd_z = L2 * sin(theta[1]) + L3 * sin(theta[1] - theta[2]) + r * sin(theta[1] - theta[2] - theta[3])
    return fwd_x, fwd_y, fwd_z


theta[0] = np.arctan2(y, x)  # Calculate initial theta[0]

def calculate_angles():
    """Find angles to match the given x, y, z."""
    for j in range(limit_theta[1]):
        for k in range(limit_theta[2]):
            theta[1] = np.deg2rad(j)
            theta[2] = np.deg2rad(k)
            fwd_x, fwd_y, fwd_z = calculate_forward_kinematics(theta)
            if np.allclose([fwd_x, fwd_y, fwd_z], [x, y, z], atol=1):
                theta[0] = np.rad2deg(theta[0])
                theta[1] = j
                theta[2] = k
                theta[3] = np.rad2deg(theta[3])
                print(f"\n{round(theta[0])},{round(theta[1])},{round(theta[2])},{round(theta[3])}")
                return True
    return False

# Check for defected areas
def constrained_area_check():
    for j in range(limit_theta[2]):
        for i in range(limit_theta[3]):
            theta[2], theta[3] = np.deg2rad(j), np.deg2rad(i)
            fwd_x, fwd_y, fwd_z = calculate_forward_kinematics(theta)
            if np.allclose([fwd_x, fwd_y, fwd_z], [x, y, z], atol=1):
                theta[:4] = map(np.rad2deg, theta[:4])
                print(f"\n{round(theta[0])},{round(theta[1])},{round(theta[2])},{round(theta[3])}")
                return True
    return False

# Evaluate conditions and find appropriate angles
if 5 <= r1 <= 17 and -8 <= z <= 15:
    theta[1] = np.deg2rad(111)
    if constrained_area_check():
        sys.exit()
elif 17 < r1 <= 26 and -8 <= z <= 15:
    theta[1] = np.deg2rad(65)
    if constrained_area_check():
        sys.exit()

# Iterate over theta[3] to find suitable angles
for i in range(limit_theta[3]):
    theta[3] = np.deg2rad(i)
    a2 = 2 * L2 * L3 + 2 * r * L2 * cos(theta[3])
    b2 = -2 * r * L2 * sin(theta[3])
    c2 = x ** 2 + y ** 2 + z ** 2 - L2 ** 2 - L3 ** 2 - r ** 2 - 2 * r * L3 * cos(theta[3])

    sqr_root_2_value = a2 ** 2 + b2 ** 2 - c2 ** 2
    if sqr_root_2_value > 0:
        theta[2] = np.arctan2(c2, np.sqrt(sqr_root_2_value)) - np.arctan2(a2, b2)
    else:
        continue

    a1 = -L3 * sin(theta[2]) - r * sin(theta[2] + theta[3])
    b1 = L2 + r * cos(theta[2] + theta[3])
    c1 = z

    sqr_root_1_value = a1 ** 2 + b1 ** 2 - c1 ** 2
    if sqr_root_1_value > 0:
        theta[1] = np.arctan2(c1, np.sqrt(sqr_root_1_value)) - np.arctan2(a1, b1)
    else:
        continue

    fwd_x, fwd_y, fwd_z = calculate_forward_kinematics(theta)
    if np.allclose([fwd_x, fwd_y, fwd_z], [x, y, z], atol=1):
        if calculate_angles():
            break
