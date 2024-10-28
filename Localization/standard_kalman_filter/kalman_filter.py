import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import csv

from utils.plot import plot_covariance_ellipse

# Covariance for KF simulation
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
]) ** 2  # predict state covariance
R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

# Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 60.0  # simulation time [s]
show_animation = True
saveToFile = False  # Set to True to save data to CSV
savePlotToFile = True  # Set to True to save final plot to file

def calc_input():
    v = 1.0  # Constant speed [m/s]
    yawrate = 0.1  # Constant yaw rate [rad/s] for circular motion
    u = np.array([[v], [yawrate]])
    return u

def observation(xTrue, xd, u):
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)

    # add noise to input
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud

def motion_model(x, u):
    # Circular motion model
    v = u[0, 0]  # speed
    yawrate = u[1, 0]  # angular velocity

    # State transition
    A = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    # Motion model for circular motion
    B = np.array([[DT * np.cos(x[2, 0]), 0],
                  [DT * np.sin(x[2, 0]), 0],
                  [0.0, DT]])

    # Update state
    x = A @ x + B @ u

    return x

def observation_model(x):
    # Linear observation model
    H = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])

    z = H @ x
    return z

def kalman_filter(xEst, PEst, z, u):
    # Predict
    A = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    B = np.array([[DT * np.cos(xEst[2, 0]), 0],
                  [DT * np.sin(xEst[2, 0]), 0],
                  [0.0, DT]])

    # Prediction step
    xPred = A @ xEst + B @ u
    PPred = A @ PEst @ A.T + Q

    # Update step
    H = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])

    zPred = H @ xPred
    y = z - zPred
    S = H @ PPred @ H.T + R
    K = PPred @ H.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ H) @ PPred

    return xEst, PEst

def save_to_csv(filename, hxTrue, hxEst, hz, hxDR):
    # Reshape history data for easier export
    hxTrue = hxTrue.T  # Transpose data for easier CSV writing
    hxEst = hxEst.T
    hz = hz.T
    hxDR = hxDR.T

    # Combine data into a single array for export
    data = np.hstack((hxTrue[:, :2], hxEst[:, :2], hz[:, :2], hxDR[:, :2]))

    # Define column headers
    headers = ['True X', 'True Y', 'Est X', 'Est Y', 'Meas X', 'Meas Y', 'DR X', 'DR Y']

    # Write to CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers
        writer.writerows(data)  # Write data

def save_final_plot(filename):
    """ Save the final plot to a file. """
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

def main():
    print(__file__ + " start!!")

    time = 0.0

    # State Vector [x y yaw]'
    xEst = np.zeros((3, 1))
    xTrue = np.zeros((3, 1))
    PEst = np.eye(3)

    xDR = np.zeros((3, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))

    while SIM_TIME >= time:
        time += DT
        u = calc_input()

        xTrue, z, xDR, ud = observation(xTrue, xDR, u)

        xEst, PEst = kalman_filter(xEst, PEst, z, ud)

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event',
                                          lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], ".g", label="Measurement")
            plt.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b", label="True")
            plt.plot(hxDR[0, :].flatten(),
                     hxDR[1, :].flatten(), "-k", label="Dead Reckoning")
            plt.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), "-r", label="Kalman Filter Estimate")
            plot_covariance_ellipse(xEst[0, 0], xEst[1, 0], PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.pause(0.001)

    if saveToFile:
        save_to_csv('kalman_filter_output.csv', hxTrue, hxEst, hz, hxDR)
        print("Data saved to kalman_filter_output.csv")

    if savePlotToFile:
        save_final_plot('kalman_filter_plot.png')
        print("Plot saved to kalman_filter_plot.png")


if __name__ == '__main__':
    main()
