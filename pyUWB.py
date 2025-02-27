# Linear solutution to UWB postioning per the University of Lubeck paper
import numpy as np
import matplotlib.pyplot as plt

# Old Anchor Positons
#A1 = [1.4732, 1.1983, 0.8890] # 1.4732, 1.1938, 0.889
#A2 = [     0,      0, 1.1684]
#A3 = [1.4732,      0, 1.1684]
#A4 = [     0, 1.2446, 1.1684]

# Position of Anchor 1
A1 = np.array([1.495, 1.25])
# Position of Anchor 2
A2 = np.array([0.05, 0.03])
# Position of Anchor 3
A3 = np.array([1.495, 0.03])
# Position of Anchor 4
A4 = np.array([0.05, 1.25])



def read_log(filename: str):
    measurements = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        r1, r2, r3, r4 = line.split(',')
        measurements.append((float(r1), float(r2), float(r3), float(r4)))
    return measurements


def read_data(filename: str):
    measurements = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        r1, r2, r3, r4, *_ = line.split(',')
        measurements.append((float(r1), float(r2), float(r3), float(r4)))

    return measurements


class UWBArray:

    def __init__(self, anchors=None):

        self._anchors = anchors

        if anchors is None:
            self._A = None
            self._K = None
        else:
            self._num_anchors = len(anchors)
            self._dim = len(anchors[0])
            if self._num_anchors <= self._dim:
                raise AttributeError(f"Anchor Array Underdefined ({len(anchors)}), at least {self._dim+1} required")
            if any([not (len(anchor) == len(anchors[0])) for anchor in anchors]):
                raise AttributeError("All Anchors must have the same dimensions")

            else:
                self._A = np.zeros((self._num_anchors-1, self._dim), dtype=float)
                self._K = np.zeros((self._num_anchors, ), dtype=float)
                for i in range(self._num_anchors-1):
                    for j in range(self._dim):
                        self._A[i][j] = 2*(self._anchors[i + 1][j] - self._anchors[0][j])
                for i in range(self._num_anchors):
                    for j in range(self._dim):
                        self._K[i] += self._anchors[i][j]**2

            self._Ainv = np.linalg.pinv(self._A)

        return

    def compute_position(self, ranges):

        if self._Ainv is None:
            raise AttributeError("No Anchor Data Found")

        if len(ranges) != self._num_anchors:
            raise ValueError(f"Inccorect Num Ranges Supplied ({len(ranges)}), {self._num_anchors} required")

        b = np.zeros((len(ranges)-1,), dtype=float)
        for i in range(1, (len(ranges))):
            b[i-1] = ranges[0]**2 - ranges[i]**2 - self._K[0] + self._K[i]
        r = self._Ainv @ b
        return r


def main():
    vectors = read_data("TraingingData/Modified/SquareTracking5.csv")
    F1_0 = 0.0254 * np.array([20.5, 20.5, 1.125])
    print(0.0254*20.5)

    anks = (A1, A2, A3, A4)
    uwbs = UWBArray(anks)
    xs = []
    ys = []
    for range_vector in vectors:
        p = uwbs.compute_position(range_vector)
        xs.append(p[0])
        ys.append(p[1])

    plt.plot(xs, ys)
    plt.show()
    print(np.mean(xs))
    print(np.mean(ys))
    return


if __name__ == "__main__":
    main()

