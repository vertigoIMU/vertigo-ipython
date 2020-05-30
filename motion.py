import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy import signal, integrate


class MotionCapture(object):
    COLUMNS = ['timestamp', 'type', 'field0', 'field1', 'field2',
               'field3', 'field4', 'field5', 'field6', 'field7']

    def __init__(self, filename):
        self.df = pd.read_csv(filename, names=self.COLUMNS, parse_dates=[0])

    @property
    def _imu_data(self):
        return self.df[self.df['type'] == 2].rename(columns={
                                                    'field0': 'ax',
                                                    'field1': 'ay',
                                                    'field2': 'az',
                                                    'field3': 'rx',
                                                    'field4': 'ry',
                                                    'field5': 'rz'
                                                    }) \
                                                    .drop(columns=['field6',
                                                                   'field7']) \
                                                    .assign(ax=lambda r: r.ax *
                                                            9.81,
                                                            ay=lambda r: r.ay *
                                                            9.81,
                                                            az=lambda r: r.az *
                                                            9.81)

    @property
    def _quat_data(self):
        return self.df[self.df['type'] == 3].rename(columns={
                                                    'field0': 'q0',
                                                    'field1': 'q1',
                                                    'field2': 'q2',
                                                    'field3': 'q3'}).drop(
                                                             columns=['field4',
                                                                      'field5',
                                                                      'field6',
                                                                      'field7'])

    @property
    def imu(self):
        r = self._quat_data.set_index('timestamp') \
                .reindex(self._imu_data.set_index('timestamp').index,
                         method='nearest').reset_index()
        return pd.merge(self._imu_data, r, on='timestamp')

    def _hamiltonian_product(self, q, r):
        return np.array([
            q[0]*r[0] - q[1]*r[1] - q[2]*r[2] - q[3]*r[3],
            q[0]*r[1] + r[0]*q[1] + q[2]*r[3] - q[3]*r[2],
            q[0]*r[2] + r[0]*q[2] + q[3]*r[1] - q[1]*r[3],
            q[0]*r[3] + r[0]*q[3] + q[1]*r[2] - q[2]*r[1]
        ])

    def _quat_conj(self, q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def _rotate(self, q, r):
        r = np.insert(r, 0, 0.0)
        o = self._hamiltonian_product(self._hamiltonian_product(q, r),
                                      self._quat_conj(q))
        return o[1:]

    def _rotate_a(self, r):
        q = np.array([r['q0'], r['q1'], r['q2'], r['q3']])
        v = [r['ax'], r['ay'], r['az']]
        return self._rotate(q, v)

    def _rotate_r(self, r):
        q = np.array([r['q0'], r['q1'], r['q2'], r['q3']])
        v = [r['rx'], r['ry'], r['rz']]
        return self._rotate(q, v)

    @property
    def board_frame(self):
        return self.imu

    @property
    def world_frame(self):
        wf = self.imu.copy()
        wf[['ax', 'ay', 'az']] = self.imu.apply(lambda x: self._rotate_a(x),
                                                axis=1,
                                                result_type='expand')
        wf[['rx', 'ry', 'rz']] = self.imu.apply(lambda x: self._rotate_r(x),
                                                axis=1,
                                                result_type='expand')
        return wf

    @property
    def linear_accel_world(self):
        return self.world_frame.assign(az=lambda r: r.az - 9.81)

    @property
    def linear_accel_board(self):
        raise NotImplementedError


if __name__ == '__main__':
    filename = 'vtg_log4.csv'
    capture = MotionCapture(filename)
    print(capture.linear_accel_world)
