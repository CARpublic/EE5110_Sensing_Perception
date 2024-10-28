import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.signal import StateSpace, lsim

# Parameters
m = 1
M = 5
L = 2
g = -10
d = 1
s = -1  # pendulum up (s = 1)

# State-space matrices
A = np.array([[0, 1, 0, 0],
              [0, -d/M, -m*g/M, 0],
              [0, 0, 0, 1],
              [0, -s*d/(M*L), -s*(m+M)*g/(M*L), 0]])

B = np.array([[0],
              [1/M],
              [0],
              [s*1/(M*L)]])

C = np.array([[1, 0, 0, 0]])

D = np.zeros((C.shape[0], B.shape[1]))

# Augment system with disturbances and noise
Vd = 0.1 * np.eye(4)  # disturbance covariance
Vn = 2  # noise covariance

BF = np.hstack([B, Vd, np.zeros_like(B)])  # augment inputs to include disturbance and noise

# Build big state space system... with single output
sysC = StateSpace(A, BF, C, np.zeros((C.shape[0], BF.shape[1])))

# System with full state output, disturbance, no noise
sysFullOutput = StateSpace(A, BF, np.eye(4), np.zeros((4, BF.shape[1])))

# Build Kalman filter
P = solve_continuous_are(A.T, C.T, Vd, Vn)  # Solve Riccati equation
L = P @ C.T @ np.linalg.inv(C @ P @ C.T + Vn)  # Kalman gain
Kf = solve_continuous_are(A, B, Vd, Vn).T  # Kalman filter design using "LQR" method

# Kalman filter state-space system
A_kf = A - L @ C
B_kf = np.hstack([B, L])
sysKF = StateSpace(A_kf, B_kf, np.eye(4), np.zeros((4, B_kf.shape[1])))

# Simulate linearized system in "down" position
dt = 0.01
t = np.arange(dt, 50, dt)

# Disturbance and noise inputs
uDIST = np.random.randn(4, len(t))
uNOISE = np.random.randn(len(t))
u = np.zeros(len(t))

# Impulse inputs
u[100:120] = 100
u[1500:1520] = -100



# Augmented input
uAUG = np.vstack([u, Vd @ uDIST, uNOISE])

# Simulate the system response
_, y, _ = lsim(sysC, U=uAUG.T, T=t)
_, xtrue, _ = lsim(sysFullOutput, U=uAUG.T, T=t)

# Simulate the Kalman filter response
_, x, _ = lsim(sysKF, U=np.vstack([u, y.T]).T, T=t)

plt.figure()
plt.plot(t, y+uNOISE, 'b', label='Measurement Cart Position')
plt.plot(t, xtrue[:, 0], 'r', label='True Cart Position')
plt.plot(t, x[:, 0], 'k--', label='Estimated Cart Position')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.title('Measurements and State Estimates')
plt.legend()

# Plot results
plt.figure()
plt.plot(t, xtrue, '-', label='True States')
plt.plot(t, x, '--', label='Estimated States [x xdot theta thetadot]')
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.title('True vs Estimated States')
plt.legend()

plt.show()
