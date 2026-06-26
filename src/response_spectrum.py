__author__ = "Boni Yogatama"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d


EC8_PARAM_TYPE_1 = pd.DataFrame(
    {
        "Ground type": ["A", "B", "C", "D", "E"],
        "S": [1, 1.2, 1.15, 1.35, 1.4],
        "T_B": [0.15, 0.15, 0.2, 0.2, 0.15],
        "T_C": [0.4, 0.5, 0.6, 0.8, 0.5],
        "T_D": [2, 2, 2, 2, 2],
    }
)

EC8_PARAM_TYPE_2 = pd.DataFrame(
    {
        "Ground type": ["A", "B", "C", "D", "E"],
        "S": [1, 1.35, 1.5, 1.8, 1.6],
        "T_B": [0.05, 0.05, 0.1, 0.1, 0.05],
        "T_C": [0.25, 0.25, 0.25, 0.3, 0.25],
        "T_D": [1.2, 1.2, 1.2, 1.2, 1.2],
    }
)


class ResponseSpectrum:
    def __init__(self):
        self.period = None
        self.spectral_acceleration = None

    def set_data(
        self,
        period: list,
        spectral_acceleration: list,
    ):
        self.period = period
        self.spectral_acceleration = spectral_acceleration

    def calculate_spectral_acceleration(self, period: float):
        if self.period is None or self.spectral_acceleration is None:
            raise ValueError("No period or spectral acceleration data.")

        FUNCTION = interp1d(
            self.period,
            self.spectral_acceleration,
            kind="linear",
        )

        SPECTRAL_ACC = FUNCTION(period)

        return SPECTRAL_ACC.tolist()

    @property
    def plot_spectrum(self):
        if self.period is None or self.spectral_acceleration is None:
            raise ValueError("No period or spectral acceleration data.")

        fig = plt.figure(figsize=(5, 4))
        plt.plot(self.period, self.spectral_acceleration, color="red")
        plt.xlim(0, 4.0)
        plt.ylim(0, np.max(self.spectral_acceleration) + 0.2)
        plt.xlabel("period (s)")
        plt.ylabel("spectral acceleration (g)")
        fig.show(False)

    @property
    def table(self):
        if self.period is None or self.spectral_acceleration is None:
            raise ValueError("No period or spectral acceleration data.")

        df = pd.DataFrame(
            {
                "T [s]": self.period,
                "Sa [g]": self.spectral_acceleration,
            }
        )

        return df


class Eurocode8(ResponseSpectrum):
    def __init__(
        self,
        ground_type: str,
        a_g: float,
        spectrum_type: int = None,
        magnitude: float = None,
    ):
        self.spectrum_type = spectrum_type
        self.magnitude = magnitude
        self.ground_type = ground_type
        self.a_g = a_g
        self.period = None
        self.spectral_acceleration = None

        if spectrum_type is None:
            if magnitude >= 5.5:
                self.spectrum_type = 1
            elif magnitude < 5.5:
                self.spectrum_type = 2

    def generate_spectrum(self):
        a_g = self.a_g

        if self.spectrum_type == 1:
            PARAM = EC8_PARAM_TYPE_1
        elif self.spectrum_type == 2:
            PARAM = EC8_PARAM_TYPE_2

        PARAM = PARAM[PARAM["Ground type"] == self.ground_type]
        S = PARAM["S"].tolist()[0]
        T_B = PARAM["T_B"].tolist()[0]
        T_C = PARAM["T_C"].tolist()[0]
        T_D = PARAM["T_D"].tolist()[0]

        T = [0, T_B, T_C]
        SA = [S * a_g, 2.5 * S * a_g, 2.5 * S * a_g]

        tail = np.linspace(T_C, 4.0)
        for t in tail:
            T.append(t)
            if t <= T_D:
                SA.append(S * a_g * 2.5 * (T_C / t))
            elif t > T_D:
                SA.append(S * a_g * 2.5 * (T_C * T_D / t**2))

        self.period = T
        self.spectral_acceleration = SA
