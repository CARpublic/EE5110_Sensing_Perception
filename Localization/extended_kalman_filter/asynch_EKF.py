import sys
import pathlib
import math
import matplotlib.pyplot as plt
import numpy as np
from utils.plot import plot_covariance_ellipse
import random

# Covariance for EKF simulation
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance
R_GPS = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance for GPS
R_IMU = np.diag([np.deg2rad(1.0), 0.1]) ** 2  # IMU covariance (yaw rate, acceleration)

# Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2
IMU_NOISE = np.diag([np.deg2rad(0.5), 0.2]) ** 2  # IMU noise

DT = 0.1  # time tick [s]
SIM_TIME = 60.0  # simulation time [s]
GPS_UPDATE_INTERVAL = 1.0  # GPS sends data every 1 second
IMU_UPDATE_INTERVAL = 0.005  # IMU sends data every 0.05 seconds

show_animation = True


def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v], [yawrate]])
    return u


def simulate_imu(xTrue, u):
    """
    Simulate IMU data (yaw rate and acceleration)
    """
    yawrate = u[1, 0]  # angular velocity (yaw rate)
    acceleration = 0.0  # for simplicity, we assume constant velocity in this simulation
    imu_data = np.array([[yawrate], [acceleration]]) + IMU_NOISE @ np.random.randn(2, 1)
    return imu_data


def simulate_gps(xTrue):
    """
    Simulate GPS data (position)
    """
    gps_data = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)
    return gps_data


def motion_model(x, u):
    """
    Car motion model with constant velocity and yaw rate
    """
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F @ x + B @ u
    return x


def observation_model(x):
    """
    GPS Observation model (x, y)
    """
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    z = H @ x
    return z


def jacob_f(x, u):
    """
    Jacobian of Motion Model
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])
    return jF


def jacob_h():
    # Jacobian of GPS Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    return jH


def ekf_prediction(xEst, PEst, u):
    """
    EKF prediction step (based on motion model and IMU input)
    """
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q
    return xPred, PPred


def ekf_update_gps(xPred, PPred, z):
    """
    EKF update step for GPS data
    """
    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R_GPS
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    return xEst, PEst


def ekf_update_imu(xPred, PPred, imu_data):
    """
    EKF update step for IMU data (yaw rate and acceleration)
    """
    # IMU only affects yaw (yaw rate) and velocity (acceleration)
    H_IMU = np.array([[0, 0, 1, 0],  # yaw rate (affects yaw)
                      [0, 0, 0, 1]])  # acceleration (affects velocity)

    zPred = H_IMU @ xPred  # IMU observation model
    y = imu_data - zPred
    S = H_IMU @ PPred @ H_IMU.T + R_IMU
    K = PPred @ H_IMU.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ H_IMU) @ PPred
    return xEst, PEst


def main():
    print(__file__ + " start!!")

    time = 0.0
    gps_time = 0.0
    imu_time = 0.0

    # State Vector [x, y, yaw, v]'
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)

    xDR = np.zeros((4, 1))  # Dead reckoning

    # Initialize GPS data with default values (to avoid accessing uninitialized variable)
    gps_data = np.zeros((2, 1))

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = gps_data  # Initialize GPS history

    while SIM_TIME >= time:
        time += DT
        gps_time += DT
        imu_time += DT

        u = calc_input()

        # Simulate car's true motion
        xTrue = motion_model(xTrue, u)

        # Simulate asynchronous IMU input (every 0.05s)
        if imu_time >= IMU_UPDATE_INTERVAL:
            imu_time = 0.0
            imu_data = simulate_imu(xTrue, u)
            xPred, PPred = ekf_prediction(xEst, PEst, u)
            xEst, PEst = ekf_update_imu(xPred, PPred, imu_data)

        # Simulate asynchronous GPS input (every 1.0s)
        if gps_time >= GPS_UPDATE_INTERVAL:
            gps_time = 0.0
            gps_data = simulate_gps(xTrue)
            xPred, PPred = ekf_prediction(xEst, PEst, u)
            xEst, PEst = ekf_update_gps(xPred, PPred, gps_data)

        # Dead reckoning (for comparison)
        xDR = motion_model(xDR, u)

        # Store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, gps_data))

        if show_animation:
            plt.cla()
            plt.plot(hz[0, :], hz[1, :], ".g", label="GPS")
            plt.plot(hxTrue[0, :].flatten(), hxTrue[1, :].flatten(), "-b", label="True")
            plt.plot(hxDR[0, :].flatten(), hxDR[1, :].flatten(), "-k", label="Dead Reckoning")
            plt.plot(hxEst[0, :].flatten(), hxEst[1, :].flatten(), "-r", label="EKF")
            plot_covariance_ellipse(xEst[0, 0], xEst[1, 0], PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.pause(0.001)


if __name__ == '__main__':
    main()
