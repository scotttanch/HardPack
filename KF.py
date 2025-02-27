import numpy as np
import matplotlib.pyplot as plt
import similaritymeasures
import copy
import time
import os
from tqdm import tqdm
"""
Extended Kalman Filter with 4 range measurements and 3 position, 3 velocity 3 linear accelerations
With variable range sensor noise

Note: Measurements and handled in lists and numpy arrays and are mutated everywhere. Just something to be aware of
"""

""" -------------------------- Validation ----------------------- """
x_init = 0.5334     # intial x-position 21-in -> 0.5334 m
y_init = 0.4895     # intial y-position 19.25-in -> 0.48895 m
z_init = 0.0254     # intial z-position 1-in -> 0.0254 m

# Square Ground Truth
sxt = [var + x_init
       for var in [0, 0, 0.127, 0.127, -0.127, -0.127, 0, 0]]          # 'Ground Truth' x-cordinates
syt = [var + y_init
       for var in [0, 0.127, 0.127, -0.127, -0.127, 0.127, 0.127, 0]]          # 'Ground Truth' y-cordinates
szt = [z_init
       for _ in sxt]          # 'Ground Truth' z-cordintates

# ZZ Ground Truth
zxt = [var + x_init
       for var in [0, 0.075, -0.075, 0.075, -0.075, 0.075, -0.075, 0.075, 0]]
zyt = [var + y_init
       for var in [0, 0, 0.15, 0.15, 0, -0.15, -0.15, 0, 0]]
zzt = [z_init
       for _ in zxt]

""" -------------------------- Helper Functions ----------------------- """


def interpolate_domain(xks, yks, zks, resolution):
    """
    Given some set of points creates a second set containing the orignal as well as points inbetween at some resolution.
    The disatance between each point may not be exactly the resolution due to the need to include the orinal points
    :param list[float] xks: Set of inital X points
    :param list[float] yks: Set of inital Y points
    :param list[float] zks: Set of inital Z points
    :param float resolution: Distance between new points
    :return: tuple containing new x, y, and z lists
    """
    # Need to create a set of domain points
    internal_x = []
    internal_y = []
    internal_z = []

    # i goes from 0 to the number of points - 1 so the last point won't be used
    # so step i will give the points in the interval between antenna position i and i+1
    for i in range(len(xks) - 1):
        # find the distance between the two points and multiply by the resolution to get the number needed
        distance = np.sqrt(
            (xks[i] - xks[i + 1]) ** 2 +
            (yks[i] - yks[i + 1]) ** 2 +
            (zks[i] - zks[i + 1]) ** 2
        )

        num_points = round(distance / resolution)
        if num_points > 0:
            # generate the sub domains but dont use the end point because it will be the start of the next
            x_sub = np.linspace(xks[i], xks[i + 1], num_points, endpoint=False)
            y_sub = np.linspace(yks[i], yks[i + 1], num_points, endpoint=False)
            z_sub = np.linspace(zks[i], zks[i + 1], num_points, endpoint=False)
            internal_x.extend(x_sub)
            internal_y.extend(y_sub)
            internal_z.extend(z_sub)

        else:
            internal_x.append(xks[i])
            internal_y.append(yks[i])
            internal_z.append(zks[i])

    # add the end point on because it won't be included in the loop

    internal_x.append(xks[-1])
    internal_y.append(yks[-1])
    internal_z.append(zks[-1])

    return internal_x, internal_y, internal_z


# Lets not use this when running the quality test, it shouldn't be nessicary
def reduce_resolution(xs: list[float], ys: list[float], zs: list[float] | None = None, resolution: float = 0.0, mean: bool = False,
                      endpoint: bool = True):
    """
    # TODO: Finish this docstring
    Given a set of points with some
    :param xs:
    :param ys:
    :param zs:
    :param resolution:
    :param mean:
    :param endpoint:
    :return:
    """

    if not zs:
        zs = np.zeros_like(xs).tolist()

    # Raise a value error is our lists arent of equal length
    if not (len(xs) == len(ys) == len(zs)):
        raise ValueError("Lists must be of equal length")

    new_x = []
    new_y = []
    new_z = []

    # This is just so pycharm will shut up about using a variable before I made it
    x_end = 0
    y_end = 0
    z_end = 0

    if endpoint:
        x_end = xs[-1]
        y_end = ys[-1]
        z_end = zs[-1]

    # In mean mode, we average points within the resolution
    if mean:
        while xs:
            _x0 = xs.pop(0)  # Pull the first point from the stacks
            y0 = ys.pop(0)
            z0 = zs.pop(0)
            x_in = [_x0]
            y_in = [y0]
            z_in = [z0]
            while True and xs:  # Make sure we dont try and pop more than we can
                d = np.sqrt((xs[0] - _x0) ** 2 + (ys[0] - y0) ** 2 + (zs[0] - z0) ** 2)
                if d < resolution:
                    x_in.append(xs.pop(0))
                    y_in.append(ys.pop(0))
                    z_in.append(zs.pop(0))
                else:
                    break

            new_x.append(np.mean(x_in))
            new_y.append(np.mean(y_in))
            new_z.append(np.mean(z_in))

    # In non-mean mode we discard points within the resolutution radius, this ensures that the returned points
    # are a subset of the original points and not a new set entirely
    if not mean:
        while xs:
            _x0 = xs.pop(0)  # Pull the first point from the stacks
            y0 = ys.pop(0)
            z0 = zs.pop(0)
            new_x.append(_x0)
            new_y.append(y0)
            new_z.append(z0)
            while xs:  # Make sure we dont try and pop more than we can
                d = np.sqrt((xs[0] - _x0) ** 2 + (ys[0] - y0) ** 2 + (zs[0] - z0) ** 2)
                if d < resolution:
                    xs.pop(0)  # Pop and discard the point at the head
                    ys.pop(0)  # Since it is in the radius we dont care about it
                    zs.pop(0)  # except we might care about it if, if its the end point
                else:
                    break

    # If we care about the end point and thus saved it earlier, add the end point if needed
    if endpoint:
        if (new_x[-1] != x_end) and (new_y[-1] != y_end) and (new_z != z_end):
            new_x.append(x_end)
            new_y.append(y_end)
            new_z.append(z_end)

    return new_x, new_y, new_z


class KalmanFilter:

    # Class Wide UWB Properties
    # Anchors
    # Position of Anchor 1
    A1 = np.array([1.495, 1.25, 0.07])
    # Position of Anchor 2
    A2 = np.array([0.05, 0.03, 0.07])
    # Position of Anchor 3
    A3 = np.array([1.495, 0.03, 0.08])
    # Position of Anchor 4
    A4 = np.array([0.05, 1.25, 0.08])
    # Noise
    rms_UWB = 0.01

    # Class Wide Realsense Properties
    rms_p = 0.0003       # according to stationary test should be 0.0003
    rms_v = 0.004       # according to stationary test should be 0.004
    rms_a = 0.05       # according to stationary test should be 0.05

    def __init__(self, intial_state=None, p_init=None, q_init=None, r_init=None, uwb_age=0, ground_truth=None, use_pose=False, use_vel=False):
        """
        Initialize a Constant Accleration 9 state Kalman Filter
        :param np.ndarray intial_state: intial state ordered as p_i, v_i, a_i
        :param float | int p_init: initial state estiamte covariance
        :param float | int q_init: state noise covariance value
        :param float | int r_init: sensor noise covariance value
        :param float | int uwb_age: maximum age of UWB measurements in ms
        :param list[list[float]] ground_truth: x, y, z cordinates of ground truth
        """
        # Initialize State
        if intial_state is not None:
            self.state = intial_state
            self.initial_state = intial_state
        else:
            self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
            self.initial_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

        # Initialize state estimate to be none, used in the create_obs method
        self.state_est = None

        # Initialize State Estimate Covariance
        if p_init is not None:
            self.P = p_init * np.identity(9, dtype=float)
        else:
            self.P = 0.1 * np.identity(9, dtype=float)

        # Initialize State Covariance
        if q_init is not None:
            self.Q = q_init * np.identity(9, dtype=float)
        else:
            self.Q = np.identity(9, dtype=float)

        # Initialize Noise Covariance
        if r_init is not None:
            self.r_init = r_init
        else:
            self.r_init = None

        # Intialize Process Model
        self.F = np.identity(9, dtype=float)

        # using raw pose
        self.pose = use_pose
        self.vel = use_vel

        # Initialize helper variables
        self.t_prev = 0     # time of previous state
        self.prev_ax = 0    # x-acceleration of previous state
        self.prev_ay = 0    # y-acceleration of previous state
        self.prev_az = 0    # z-acceleration of previous state
        self.jx = 0         # maximum x-jerk
        self.jy = 0         # maximum y-jerk
        self.jz = 0         # maximum z-jerk

        # Initialize UWB Parameters
        self.AGE_LIMIT = uwb_age
        self.R_W0 = self.rms_UWB
        self.used_uwbs = []

        # Initialize History and Measruements Atributes
        self.state_history = None
        self._measurement_vectors = None

        # Initialize Verification Atributes
        self.fd = None

        if ground_truth is not None:
            self.xt = ground_truth[0]
            self.yt = ground_truth[1]
            self.zt = ground_truth[2]
        else:
            self.xt = None
            self.yt = None
            self.zt = None

        return

    def read_data(self, file_path):
        """
        Process a csv file containing measurements and stores it in the instance
        :param file_path:
        :return:
        """
        measurements = []
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            measures = line.split(',')
            measurements.append([float(each) for each in measures])

        # ordered as r1, r2, r3, r4, a1, a2, a3, a4, rx, ry, rz, vx, vy, vz, ax, ay, az, t
        self._measurement_vectors = measurements

        return

    def retrieve_data(self):
        """
        Returns a copy of the raw measurement data
        :return:
        """
        if self._measurement_vectors is not None:
            return copy.deepcopy(self._measurement_vectors)
        else:
            raise AttributeError("No Measurements found")

    def _create_obs_(self, raw_measurement):
        """
        Creates an observation model based on a raw measurement vector
        :param list[float] raw_measurement: raw measurement vector ordered as r[1-4], a[1-4], px, py, pz, vx, vy, vz, ax, ay, az
        :return:
        """
        # make a copy of the measurment list, beacuse I dont want to get into trouble with weird mutation stuff
        _mod_measurement = raw_measurement.copy()

        # pop out the relevant ages
        _a1 = _mod_measurement.pop(4)
        _a2 = _mod_measurement.pop(4)
        _a3 = _mod_measurement.pop(4)
        _a4 = _mod_measurement.pop(4)

        _h_tail = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # px 4
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # py 5
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],      # pz 6
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # vx 7
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],      # vy 8
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],      # vz 9
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # ax 10
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],      # ay 11
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])     # az 12

        _w_tail = np.array([self.rms_p, self.rms_p, self.rms_p, self.rms_v, self.rms_v, self.rms_v, self.rms_a, self.rms_a, self.rms_a])

        if not self.pose:

            # When not using pose, toss out the position data and trim _h and _w to remove p and v
            _mod_measurement.pop(4)
            _mod_measurement.pop(4)
            _mod_measurement.pop(4)

            _h_tail = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # vx 7
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # vy 8
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # vz 9
                                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ax 10
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # ay 11
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]) # az 12

            _w_tail = np.array([ self.rms_v, self.rms_v, self.rms_v, self.rms_a, self.rms_a, self.rms_a])

        # Considering using the maximum jerk to add noise to the uwb?
        _j = np.sqrt(self.jx**2 + self.jy**2 + self.jz**2)

        # I want to keep track of which if any uwbs get used
        _used = set()

        # If the age of the measurement is less than the limit and its not nan we add its row to the observation model
        if _a4 < self.AGE_LIMIT and not np.isnan(_mod_measurement[3]):
            _dr4_dx = (self.state_est[0] - self.A4[0]) / np.sqrt((self.state_est[0] - self.A4[0]) ** 2 + (self.state_est[3] - self.A4[1]) ** 2 + (self.state_est[6] - self.A4[2]) ** 2)
            _dr4_dy = (self.state_est[3] - self.A4[1]) / np.sqrt((self.state_est[0] - self.A4[0]) ** 2 + (self.state_est[3] - self.A4[1]) ** 2 + (self.state_est[6] - self.A4[2]) ** 2)
            _dr4_dz = (self.state_est[6] - self.A4[2]) / np.sqrt((self.state_est[0] - self.A4[0]) ** 2 + (self.state_est[3] - self.A4[1]) ** 2 + (self.state_est[6] - self.A4[2]) ** 2)

            # stack this row on top of the tail
            _h_tail = np.vstack((np.array([_dr4_dx, 0, 0, _dr4_dy, 0, 0, _dr4_dz, 0, 0]), _h_tail))

            # Add the noise to the noise tail, increasing it by the distance indicated by the 1/6*jerk*age**3
            _w_tail = np.hstack((np.array([self.rms_UWB + (1/6)*(_a4 * (10**-3))**3 * _j]), _w_tail))
            noise = self.rms_UWB + (1/6)*(_a4 * (10**-3))**3 * _j
            #print(self.rms_UWB + (1/6)*(_a4 * (10**-3))**3 * _j)
            # add 4 to the used track
            # adjust for mean error
            _mod_measurement[3] = _mod_measurement[3] - 0.18
            _used.add(4)
        else:
            _mod_measurement[3] = np.nan

        if _a3 < self.AGE_LIMIT and not np.isnan(_mod_measurement[2]):
            _dr3_dx = (self.state_est[0] - self.A3[0]) / np.sqrt((self.state_est[0] - self.A3[0]) ** 2 + (self.state_est[3] - self.A3[1]) ** 2 + (self.state_est[6] - self.A3[2]) ** 2)
            _dr3_dy = (self.state_est[3] - self.A3[1]) / np.sqrt((self.state_est[0] - self.A3[0]) ** 2 + (self.state_est[3] - self.A3[1]) ** 2 + (self.state_est[6] - self.A3[2]) ** 2)
            _dr3_dz = (self.state_est[6] - self.A3[2]) / np.sqrt((self.state_est[0] - self.A3[0]) ** 2 + (self.state_est[3] - self.A3[1]) ** 2 + (self.state_est[6] - self.A3[2]) ** 2)

            # stack this row on top of the tail
            _h_tail = np.vstack((np.array([_dr3_dx, 0, 0, _dr3_dy, 0, 0, _dr3_dz, 0, 0]), _h_tail))

            # add the appropriate noise to the noise tail
            _w_tail = np.hstack((np.array([self.rms_UWB + (1/6)*(_a3 * (10**-3))**3 * _j]), _w_tail))

            _mod_measurement[2] = _mod_measurement[2] - 0.67
            _used.add(3)
        else:
            _mod_measurement[2] = np.nan

        if _a2 < self.AGE_LIMIT and not np.isnan(_mod_measurement[1]):
            _dr2_dx = (self.state_est[0] - self.A2[0]) / np.sqrt((self.state_est[0] - self.A2[0]) ** 2 + (self.state_est[3] - self.A2[1]) ** 2 + (self.state_est[6] - self.A2[2]) ** 2)
            _dr2_dy = (self.state_est[3] - self.A2[1]) / np.sqrt((self.state_est[0] - self.A2[0]) ** 2 + (self.state_est[3] - self.A2[1]) ** 2 + (self.state_est[6] - self.A2[2]) ** 2)
            _dr2_dz = (self.state_est[6] - self.A2[2]) / np.sqrt((self.state_est[0] - self.A2[0]) ** 2 + (self.state_est[3] - self.A2[1]) ** 2 + (self.state_est[6] - self.A2[2]) ** 2)

            _h_tail = np.vstack((np.array([_dr2_dx, 0, 0, _dr2_dy, 0, 0, _dr2_dz, 0, 0]), _h_tail))
            _w_tail = np.hstack((np.array([self.rms_UWB + (1/6)*(_a2 * (10**-3))**3 * _j]), _w_tail))

            _mod_measurement[1] = _mod_measurement[1] - 0.56

            _used.add(2)

        else:
            _mod_measurement[1] = np.nan

        if _a1 < self.AGE_LIMIT and not np.isnan(_mod_measurement[0]):
            _dr1_dx = (self.state_est[0] - self.A1[0]) / np.sqrt((self.state_est[0] - self.A1[0]) ** 2 + (self.state_est[3] - self.A1[1]) ** 2 + (self.state_est[6] - self.A1[2]) ** 2)
            _dr1_dy = (self.state_est[3] - self.A1[1]) / np.sqrt((self.state_est[0] - self.A1[0]) ** 2 + (self.state_est[3] - self.A1[1]) ** 2 + (self.state_est[6] - self.A1[2]) ** 2)
            _dr1_dz = (self.state_est[6] - self.A1[2]) / np.sqrt((self.state_est[0] - self.A1[0]) ** 2 + (self.state_est[3] - self.A1[1]) ** 2 + (self.state_est[6] - self.A1[2]) ** 2)

            _h_tail = np.vstack((np.array([_dr1_dx, 0, 0, _dr1_dy, 0, 0, _dr1_dz, 0, 0]), _h_tail))
            _w_tail = np.hstack((np.array([self.rms_UWB + (1/6)*(_a1 * (10 ** -3))**3 * _j]), _w_tail))

            _mod_measurement[0] = _mod_measurement[0] - 0.41

            _used.add(1)
        else:
            _mod_measurement[0] = np.nan

        # filter out any nans in the measurment list so we only have measurements for rows we added to the obs model
        _mod_measurement = [item for item in _mod_measurement if (not np.isnan(item))]
        return np.array(_mod_measurement), _h_tail, _w_tail, _used

    def update_state(self, raw_measurement):
        """
        Update the current state based on some measurement vector
        :param list raw_measurement: raw measurement vector ordered as r[1-4], a[1-4], px, py, pz, vx, vy, vz, ax, ay, az, t
        :return:
        """
        # Step 1: pull the time off the measurment vector and update dt
        _time = raw_measurement.pop()
        _dt = _time - self.t_prev
        _adt = 0.5 * (_dt ** 2)

        # Step 2: update the Plant Process Model using the current time step
        self.F[0][1] = _dt
        self.F[1][2] = _dt
        self.F[0][2] = _adt

        self.F[3][4] = _dt
        self.F[4][5] = _dt
        self.F[3][5] = _adt

        self.F[6][7] = _dt
        self.F[7][8] = _dt
        self.F[6][8] = _adt

        # Step 3: pull out the acclertation and update the maximum jerk
        _ax = raw_measurement[14]
        _ay = raw_measurement[15]
        _az = raw_measurement[16]

        if self.jx < abs(self.prev_ax - _ax):
            self.jx = abs(self.prev_ax - _ax)
        self.prev_ax = _ax

        if self.jy < abs(self.prev_ay - _ay):
            self.jy = abs(self.prev_ay - _ay)
        self.prev_ay = _ay

        if self.jz < abs(self.prev_az - _az):
            self.jz = abs(self.prev_az - _az)
        self.prev_az = _az

        # step 4: Adjust the realsense measurement by the inital state
        raw_measurement[8] = raw_measurement[8] + self.initial_state[0]
        raw_measurement[9] = raw_measurement[9] + self.initial_state[3]
        raw_measurement[10] = raw_measurement[10] + self.initial_state[6]

        # Process Noise (see C. Barios paper for this)
        _fn = np.array([(1 / 6) * self.jx * (_dt ** 3),
                        (1 / 2) * self.jx * (_dt ** 2),
                        self.jx * _dt,
                        (1 / 6) * self.jy * (_dt ** 3),
                        (1 / 2) * self.jy * (_dt ** 2),
                        self.jy * _dt,
                        (1 / 6) * self.jz * (_dt ** 3),
                        (1 / 2) * self.jz * (_dt ** 2),
                        self.jz * _dt])

        # Estimate new position
        self.state_est = (self.F @ self.state) + _fn

        # Step 5: Generate a revised measurement vector, observation matrix, and sensor noise vector
        _measure, _h, _w, _u = self._create_obs_(raw_measurement)

        # For tracking what uwbs got used so we can color points
        self.used_uwbs.append(_u)

        if self.r_init is not None:
            _r = self.r_init * np.identity(len(_w))
        else:
            _r = np.identity(len(_w))
            for i in range(len(_w)):
                _r[i, i] = _w[i]**2

        # Predicition Covatiance
        _pk_k1 = (self.F @ self.P @ self.F.T) + self.Q

        # Invoation
        _yk = _measure - ((_h @ self.state_est) + _w)

        # Invoation Covariance
        _sk = (_h @ _pk_k1 @ _h.T) + _r

        # Kalman Gain
        _k_k = _pk_k1 @ _h.T @ np.linalg.pinv(_sk)

        # Update state estimate
        self.state = self.state_est + (_k_k @ _yk)

        # Update State Covariance
        self.P = _pk_k1 - (_k_k @ _h @ _pk_k1)

        # Overwrite previous time step
        self.t_prev = _time

        return self.state, _time

    def calculate_states(self):
        """
        Calculate the state history based on a series of measurement vectors
        :return:
        """
        self.state_history = []
        for vector in self.retrieve_data():
            self.state_history.append(self.update_state(vector))
        return

    def plot_history(self):
        """
        Plot the state history with optional true path
        :return:
        """
        xs = []
        ys = []
        zs = []
        ts = []

        for state, t in self.state_history:
            xs.append(state[0])
            ys.append(state[3])
            zs.append(state[6])
            ts.append(t)

        fig, [ax0, ax1] = plt.subplots(1, 2)

        fig.suptitle("KF Computed Position")

        ax0.plot(xs, ys)
        ax0.scatter(xs[0], ys[0], color='red')
        ax0.scatter(xs[-1], ys[-1], color='yellow')
        if self.xt is not None:
            ax0.plot(self.xt, self.yt, color='black')
        else:
            ax0.set_title(f"XY Plane Postion")

        ax0.set_xlabel("X-Cordinate (m)")
        ax0.set_ylabel("Y-Cordinate (m)")
        ax0.set_aspect('equal')

        ax1.plot(ts, zs)
        ax1.set_title("Z Postition")
        ax1.set_ylabel("Z-Cordinate (m)")
        ax1.set_xlabel("Time (s)")

        plt.show()

        return

    def evaluate(self):
        """
        Compute performace of state history by calculating the discrete frechet distance between ground truth and
        state history
        :return:
        """

        # To evaluate performance we need ground truth
        if self.xt is None:
            raise AttributeError("No Ground Truth Found")

        i_xt, i_yt, i_zt = interpolate_domain(self.xt, self.yt, self.zt, 0.01)
        i_truth = np.transpose(np.vstack((i_xt, i_yt, i_zt)))

        xs = []
        ys = []
        zs = []
        for state, _ in self.state_history:
            xs.append(state[0])
            ys.append(state[3])
            zs.append(state[6])

        data = np.transpose(np.vstack((xs, ys, zs)))
        self.fd = similaritymeasures.frechet_dist(data, i_truth)

        return self.fd

    def compare_raw(self, include_z=False, print_only=False):
        """
        Compares filter perfomance against raw sensor data by computing the discrete frechet distance between both data
        sets and ground truth
        :param bool include_z: Include z-cordinate in caluclation of frechet distances, default False
        :param bool print_only: Skip plotting and return the performance increase
        :return: difference in meters, percent increase
        """

        # To compare performance we need ground truth
        if self.xt is None:
            raise AttributeError("No Ground Truth Found")

        # Interpolate ground truth to mm resolution, which seems to affect the freceht distance for some reason
        i_xt, i_yt, i_zt = interpolate_domain(self.xt, self.yt, self.zt, 0.01)

        # from the raw vectors we need to extract the RS position
        raw_x = []
        raw_y = []
        raw_z = []
        raw_t = []

        for vector in self.retrieve_data():
            raw_x.append(vector[8] + self.initial_state[0])
            raw_y.append(vector[9] + self.initial_state[3])
            raw_z.append(vector[10] + self.initial_state[6])
            raw_t.append(vector[17])

        if self.state_history is None:
            self.calculate_states()

        data_x = []
        data_y = []
        data_z = []
        data_t = []

        for state, t in self.state_history:
            data_x.append(state[0])
            data_y.append(state[3])
            data_z.append(state[6])
            data_t.append(t)

        data_color = []
        for each in self.used_uwbs:
            if each == set():
                data_color.append('red')
            else:
                data_color.append('blue')

        if not include_z:
            i_truth = np.transpose(np.vstack((i_xt, i_yt)))
            raw_data = np.transpose(np.vstack((raw_x, raw_y)))
            filtered_data = np.transpose(np.vstack((data_x, data_y)))
        else:
            i_truth = np.transpose(np.vstack((i_xt, i_yt, i_zt)))
            raw_data = np.transpose(np.vstack((raw_x, raw_y, raw_z)))
            filtered_data = np.transpose(np.vstack((data_x, data_y, data_z)))

        filter_fd = similaritymeasures.frechet_dist(filtered_data, i_truth)
        rs_fd = similaritymeasures.frechet_dist(raw_data, i_truth)
        dif = rs_fd - filter_fd
        perfomance_imporvement = (dif/rs_fd)*100

        if print_only:
            return dif, perfomance_imporvement

        fig, [[ax0, ax1], [ax2, ax3]] = plt.subplots(2, 2)

        fig.suptitle("Comparison of Raw Pose to Kalman Filter")

        ax0.set_aspect('equal')
        ax2.set_aspect('equal')

        ax0.set_title(f"Raw Pose \n Frechet Distance: {rs_fd*100:.4f} cm")
        ax0.plot(raw_x, raw_y, color='red')
        ax0.plot(self.xt, self.yt, color='black')
        ax0.set_xlabel("X-Cordinate (m)")
        ax0.set_ylabel("Y-Cordinate (m)")

        ax1.set_title(f"Raw Z-Cordinate")
        ax1.plot(raw_t, raw_z, color='red')
        ax1.axhline(i_zt[0], color='black')         # This line is only applicable in the cnc bench case
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Z-Cordinate (m)")

        ax2.set_title(f"Filtered Position \n Frechet Distance {filter_fd*100:.4f} cm")
        # ax2.plot(data_x, data_y, color='blue', linewidth=1)
        ax2.scatter(data_x, data_y, c=data_color, s=3)
        ax2.plot(self.xt, self.yt, color='black')
        ax2.set_xlabel("X-Cordinate (m)")
        ax2.set_ylabel("Y-Cordinate (m)")

        ax3.set_title("Filtered Z-Codrinate")
        # ax3.plot(data_t, data_z, color='blue', linewidth=1)
        ax3.scatter(data_t, data_z, c=data_color, s=3)
        ax3.axhline(i_zt[0], color='black')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Z-Cordinate (m)")

        # Adjust the lauput to how i know it should be
        fig.tight_layout(h_pad=0.2, w_pad=0.0)
        fig.subplots_adjust(left=0.07, bottom=0.08, right=0.86, top=0.89, wspace=0, hspace=0.343)
        plt.show()

        return dif, perfomance_imporvement

    def reset(self):
        """
        Resets filter to initial state
        :return:
        """
        # TODO: Figure out how to get this going without doing a ton of code duplication

    @classmethod
    def update_anchor(cls, number, cordinates):
        """
        Update the x, y, z corindates of an anchor
        :param int number:
        :param list[float, float, float] cordinates:
        :return:
        """
        if number == 1:
            cls.A1 = cordinates
        if number == 2:
            cls.A2 = cordinates
        if number == 3:
            cls.A3 = cordinates
        if number == 4:
            cls.A4 = cordinates

    @classmethod
    def change_rms_u(cls, noise):
        cls.rms_UWB = noise

    @classmethod
    def change_rms_a(cls, noise):
        cls.rms_a = noise

    @classmethod
    def change_rms_v(cls, noise):
        cls.rms_v = noise

    @classmethod
    def change_rms_p(cls, noise):
        cls.rms_p = noise

    def __repr__(self):
        rep = (f"Current State \n"
               f"Pos: ({self.state[0]:.4f},{self.state[3]:.4f},{self.state[6]:.4f}) \n"
               f"Vel: ({self.state[1]:.2f},{self.state[4]:.2f},{self.state[7]:.2f}) \n"
               f"Acc: ({self.state[2]:.2f},{self.state[5]:.2f},{self.state[8]:.2f})")
        return rep


def time_filter():
    """
    Tests filter performance by calculating the mean time to estimate a new state
    :return:
    """
    x0 = np.array([x_init, 0.0, 0.0, y_init, 0.0, 0.0, z_init, 0.0, 0.0])
    kf = KalmanFilter(p_init=0.1, q_init=0.001, uwb_age=250, intial_state=x0, ground_truth=[sxt, syt, szt])
    kf.read_data("TraingingData/Modified/Square11.csv")

    data = kf.retrieve_data()
    cum_sum = 0
    for vector in data:
        start = time.monotonic()
        kf.update_state(vector)
        cum_sum += time.monotonic() - start
    print(f"Mean Time to estimate: {(cum_sum/len(data))*1e6:.2f}\u03BCs over {len(data)} states")
    return


def train_filter(q_range, input_data, eval_path):
    """
    Train the filter by finding the values of Q_init that result in the mimimum frechet distance to between the input
    data file and some ground truth
    :param np.ndarray[float|int] q_range:
    :param str input_data:
    :param list[list[float]] eval_path:
    :return:
    """
    x0 = np.array([x_init, 0.0, 0.0, y_init, 0.0, 0.0, z_init, 0.0, 0.0])

    # Over some range of p and q, calcualte the position history and compute the frechet distance for that
    # combination of p and q
    results = np.zeros_like(q_range)
    for i in tqdm(range(len(q_range))):
        kf = KalmanFilter(p_init=0.1, q_init=q_range[i], uwb_age=250, intial_state=x0, ground_truth=eval_path)
        kf.read_data(input_data)
        results = kf.evaluate()

    # find the index of the minimum frechet distance
    ind = np.unravel_index(np.argmin(results, axis=None), results.shape)

    return results[ind], q_range[ind], results


def process_data_set(data_directory):
    """
    Iterate through all trainging data and find the perfomance increase of the filter over raw pose measurments

    Example usage:
        process_data_set('TraingingData\\Modified')
    :param data_directory:
    :return:
    """
    files = os.listdir(data_directory)
    for file in files:
        name = file.split('.')[0]
        data = os.path.join(data_directory, file)

        if name.startswith("S"):
            truth = [sxt, syt, szt]
        elif name.startswith('Z'):
            truth = [zxt, zyt, zzt]

        x0 = np.array([x_init, 0.0, 0.0, y_init, 0.0, 0.0, z_init, 0.0, 0.0])
        kf = KalmanFilter(p_init=0.1, q_init=5e-7, uwb_age=500,
                          ground_truth=truth, intial_state=x0, use_pose=True)
        kf.read_data(data)
        dif, inc = kf.compare_raw(include_z=True, print_only=True)
        print(f"{name} Performance Increase: {dif*1000:.2f}mm ({inc:.2f}%)")


def example():
    x0 = np.array([x_init, 0.0, 0.0, y_init, 0.0, 0.0, z_init, 0.0, 0.0])
    kf = KalmanFilter(p_init=0.1, q_init=5e-8, uwb_age=500, ground_truth=[sxt, syt, szt], intial_state=x0, use_pose=True)
    kf.read_data("TraingingData\\Modified\\SquareTracking1.csv")
    dif, inc = kf.compare_raw(include_z=True, print_only=False)
    print(f"Performance Increase: {inc:.2f}%")
    return


def main():
    #process_data_set('TraingingData\\Modified')
    example()
    return


if __name__ == '__main__':
    main()


""" ----- Kalman Filter Class Example Usage ----- """

"""
measures = read_data("TraingingData/Modified/Square10.csv")                 # Read In data
x0 = np.array([x_init, 0.0, 0.0, y_init, 0.0, 0.0, z_init, 0.0, 0.0])       # Provide an intial state
kf1 = KalmanFilter(p_init=0.1, q_init=0.1, intial_state=x0, uwb_age=0)      # Construct the filter
kf1.calculate_states(copy.deepcopy(measures))                               # Pass a copy of data, since the class mutates each vector
kf1.plot_history(truth=[xt, yt, zt])                                        # plot the state history, with optional ground truth
fd1 = kf1.evaluate([xt, yt, zt])                                            # compute the frechet distance to ground truth
print(fd1)
"""

"""Performance
Time per step:
    Mean Time to filter: 0.0006669175357488904
    Median Time to filter: 0.0
    Max Time to filter: 0.35900000000037835
    Min Time to filter: 0.0
"""