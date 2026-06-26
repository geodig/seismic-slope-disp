__author__ = "Boni Yogatama"

import numpy as np
import pandas as pd
from scipy.stats import norm


def prob_disp_bray(
    probability: float,
    P_D_0: float,
    median_displacement: float,
    sigma: float,
):
    P_D_d_D_0 = probability / (1 - P_D_0)
    phi_Z = 1 - P_D_d_D_0
    Z = norm.ppf(phi_Z)
    ln_d = Z * sigma + np.log(median_displacement)
    d = np.exp(ln_d)
    return d


class BrayTravasarou2007:
    def __init__(
        self,
        yield_coefficient: float,
        natural_period: float,
        magnitude: float,
        spectral_acceleration_150_pct_period: float,
    ):
        self.ky = yield_coefficient
        self.Ts = natural_period
        self.Mw = magnitude
        self.Sa_15Ts = spectral_acceleration_150_pct_period
        self.sigma = 0.66
        self.result = dict()
        self.result["non-zero_seismic_displacement [cm]"] = None
        self.result["displacement_with_probability [cm]"] = dict()

    def calculate_nonzero_seismic_displacement(self):
        ln_ky = np.log(self.ky)
        ln_Sa = np.log(self.Sa_15Ts)

        # equation 5
        ln_D = (
            -1.1
            - 2.83 * ln_ky
            - 0.333 * ln_ky**2
            + 0.566 * ln_ky * ln_Sa
            + 3.04 * ln_Sa
            - 0.244 * ln_Sa**2
            + 1.5 * self.Ts
            + 0.278 * (self.Mw - 7)
        )
        D = np.exp(ln_D)
        self.D = D
        self.result["non-zero_seismic_displacement [cm]"] = np.round(D, 2)

    def calculate_displacement_with_probability(
        self,
        probability: list,
        threshold_displacement: float,
    ):
        ln_ky = np.log(self.ky)
        ln_Sa = np.log(self.Sa_15Ts)
        d_thres = threshold_displacement

        if self.result["non-zero_seismic_displacement [cm]"] is None:
            self.calculate_nonzero_seismic_displacement()

        # equation 3
        P_D_0 = 1 - norm.cdf(
            -1.76 - 3.22 * ln_ky - 0.484 * self.Ts * ln_ky + 3.52 * ln_Sa
        )
        self.result["P(D='0')"] = np.round(P_D_0, 2)

        # equation 7 and 8
        P_D_d = (1 - P_D_0) * (
            norm.cdf((np.log(self.D) - np.log(d_thres)) / self.sigma)
        )
        self.result["P(D > d_threshold)"] = np.round(P_D_d, 2)

        list_displacement = []
        for prob in probability:
            if prob >= 1.0:
                raise ValueError("Probability should be in decimal!")
            elif prob < 0.0:
                raise ValueError("Probability should be positive!")

            try:
                # P_D_d_D_0 = prob / (1 - P_D_0)
                # phi_Z = 1 - P_D_d_D_0
                # Z = norm.ppf(phi_Z)
                # ln_d = Z * self.sigma + np.log(self.D)
                # d = np.exp(ln_d)
                d = prob_disp_bray(prob, P_D_0, self.D, self.sigma)
                list_displacement.append(d)
            except ValueError:
                list_displacement.append(np.nan)

        for i in range(len(list_displacement)):
            prob = probability[i]
            disp = np.round(list_displacement[i], 2)
            self.result["displacement_with_probability [cm]"][prob] = disp


class FotopoulosPitilakis2016:
    def __init__(
        self,
        yield_coefficient: float,
        magnitude: float,
        peak_ground_acceleration: float,
        kind: str,
    ):
        self.ky = yield_coefficient
        self.Mw = magnitude
        self.PGA = peak_ground_acceleration
        self.result = dict()
        self.kind = kind
        self.result["displacement_with_probability [cm]"] = dict()

        self.kind_options = [
            "PGA_M",
            "ky_PGA_M",
        ]

        if kind not in self.kind_options:
            raise ValueError("kind input is not valid.")

    def calculate_displacement_with_probability(
        self,
        probability: list,
    ):
        PARAM_TABLE = pd.DataFrame(
            {
                "kind": self.kind_options,
                "IM": [
                    self.PGA,
                    self.ky / self.PGA,
                ],
                "a": [
                    -2.965,
                    -10.246,
                ],
                "b": [
                    2.127,
                    -2.165,
                ],
                "c": [
                    -6.583,
                    7.844,
                ],
                "d": [
                    0.535,
                    0.654,
                ],
                "sigma": [
                    0.72,
                    0.75,
                ],
            }
        )

        kind = self.kind
        PARAM = PARAM_TABLE[PARAM_TABLE["kind"] == kind]
        IM = PARAM["IM"].tolist()[0]
        a = PARAM["a"].tolist()[0]
        b = PARAM["b"].tolist()[0]
        c = PARAM["c"].tolist()[0]
        d = PARAM["d"].tolist()[0]
        sigma = PARAM["sigma"].tolist()[0]

        ln_D = b * np.log(IM) + a + c * self.ky + d * self.Mw

        list_displacement = []
        for prob in probability:
            if prob >= 1.0:
                raise ValueError("Probability should be in decimal!")
            elif prob < 0.0:
                raise ValueError("Probability should be positive!")

            try:
                Z = norm.ppf(prob)
                ln_d = Z * sigma + ln_D
                d = (np.exp(ln_d)) * 100
                list_displacement.append(d)
            except ValueError:
                list_displacement.append(np.nan)

        for i in range(len(list_displacement)):
            prob = probability[i]
            disp = np.round(list_displacement[i], 2)
            self.result["displacement_with_probability [cm]"][prob] = disp


class BrayMacedoTravasarou2018:
    def __init__(
        self,
        yield_coefficient: float,
        natural_period: float,
        magnitude: float,
        spectral_acceleration_150_pct_period: float,
    ):
        self.ky = yield_coefficient
        self.Ts = natural_period
        self.Mw = magnitude
        self.Sa_15Ts = spectral_acceleration_150_pct_period
        self.sigma = 0.73
        self.result = dict()
        self.result["non-zero_seismic_displacement [cm]"] = None
        self.result["displacement_with_probability [cm]"] = dict()

    def calculate_nonzero_seismic_displacement(self):
        ln_ky = np.log(self.ky)
        ln_Sa = np.log(self.Sa_15Ts)

        # equation 4
        if self.Ts >= 0.1:
            a1 = -6.896
            a2 = 3.081
            a3 = -0.803
        elif self.Ts < 0.1:
            a1 = -5.864
            a2 = -9.421
            a3 = 0.0

        ln_D = (
            a1
            - 3.353 * ln_ky
            - 0.39 * ln_ky**2
            + 0.538 * ln_ky * ln_Sa
            + 3.06 * ln_Sa
            - 0.225 * ln_Sa**2
            + a2 * self.Ts
            + a3 * (self.Ts) ** 2
            + 0.55 * self.Mw
        )

        D = np.exp(ln_D)
        self.D = D
        self.result["non-zero_seismic_displacement [cm]"] = np.round(D, 2)

    def calculate_displacement_with_probability(
        self,
        probability: list,
        threshold_displacement: float,
    ):
        ln_ky = np.log(self.ky)
        ln_Sa = np.log(self.Sa_15Ts)
        d_thres = threshold_displacement

        if self.result["non-zero_seismic_displacement [cm]"] is None:
            self.calculate_nonzero_seismic_displacement()

        # equation 2 & 3
        if self.Ts <= 0.7:
            P_D_0 = 1 - norm.cdf(
                -2.64
                - 3.2 * ln_ky
                - 0.17 * ln_ky**2
                - 0.49 * self.Ts * ln_ky
                + 2.09 * self.Ts
                + 2.91 * ln_Sa
            )
        elif self.Ts > 0.7:
            P_D_0 = 1 - norm.cdf(
                -3.53
                - 4.78 * ln_ky
                - 0.34 * ln_ky**2
                - 0.3 * self.Ts * ln_ky
                - 0.67 * self.Ts
                + 2.66 * ln_Sa
            )

        self.result["P(D='0')"] = np.round(P_D_0, 2)

        # equation 6 and 7
        P_D_d = (1 - P_D_0) * (
            norm.cdf((np.log(self.D) - np.log(d_thres)) / self.sigma)
        )
        self.result["P(D > d_threshold)"] = np.round(P_D_d, 2)

        list_displacement = []
        for prob in probability:
            if prob >= 1.0:
                raise ValueError("Probability should be in decimal!")
            elif prob < 0.0:
                raise ValueError("Probability should be positive!")

            try:
                # P_D_d_D_0 = prob / (1 - P_D_0)
                # phi_Z = 1 - P_D_d_D_0
                # Z = norm.ppf(phi_Z)
                # ln_d = Z * self.sigma + np.log(self.D)
                # d = np.exp(ln_d)
                d = prob_disp_bray(prob, P_D_0, self.D, self.sigma)
                list_displacement.append(d)
            except ValueError:
                list_displacement.append(np.nan)

        for i in range(len(list_displacement)):
            prob = probability[i]
            disp = np.round(list_displacement[i], 2)
            self.result["displacement_with_probability [cm]"][prob] = disp


class BrayMacedo2019:
    def __init__(
        self,
        kind: str,
        yield_coefficient: float,
        natural_period: float,
        magnitude: float,
        spectral_acceleration_130_pct_period: float,
        peak_ground_velocity: float = None,
    ):
        self.kind = kind
        self.ky = yield_coefficient
        self.Ts = natural_period
        self.Mw = magnitude
        self.Sa_13Ts = spectral_acceleration_130_pct_period
        self.PGV = peak_ground_velocity
        self.sigma = None
        self.result = dict()
        self.result["non-zero_seismic_displacement [cm]"] = None
        self.result["displacement_with_probability [cm]"] = dict()

        if self.kind != "ordinary_gm" and peak_ground_velocity is None:
            raise ValueError("For the selected kind, PGV value is required!")

    def calculate_nonzero_seismic_displacement(self):
        ln_ky = np.log(self.ky)
        ln_Sa = np.log(self.Sa_13Ts)

        # equation 3 / 5 / 7 / 9

        if self.kind == "ordinary_gm":
            # Equation 3
            if self.Ts >= 0.1:
                a1 = -5.981
                a2 = 3.223
                a3 = -0.945
            elif self.Ts < 0.1:
                a1 = -4.684
                a2 = -9.471
                a3 = 0.0

            ln_D = (
                a1
                - 2.482 * ln_ky
                - 0.244 * ln_ky**2
                + 0.344 * ln_ky * ln_Sa
                + 2.649 * ln_Sa
                - 0.09 * ln_Sa**2
                + a2 * self.Ts
                + a3 * (self.Ts) ** 2
                + 0.603 * self.Mw
            )

        elif self.kind == "near_fault_D100_gm":
            # Equation 5
            ln_PGV = np.log(self.PGV)

            if self.PGV <= 150:
                if self.Ts >= 0.1:
                    c1 = -6.951
                    c2 = 1.069
                    c3 = -0.498
                    c4 = 1.547
                elif self.Ts < 0.1:
                    c1 = -6.724
                    c2 = -2.744
                    c3 = 0.0
                    c4 = 1.547
            elif self.PGV > 150:
                if self.Ts >= 0.1:
                    c1 = 1.764
                    c2 = 1.069
                    c3 = -0.498
                    c4 = -0.097
                elif self.Ts < 0.1:
                    c1 = 1.991
                    c2 = -2.744
                    c3 = 0.0
                    c4 = -0.097

            ln_D = (
                c1
                - 2.632 * ln_ky
                - 0.278 * ln_ky**2
                + 0.527 * ln_ky * ln_Sa
                + 1.978 * ln_Sa
                - 0.233 * ln_Sa**2
                + c2 * self.Ts
                + c3 * (self.Ts) ** 2
                + 0.01 * self.Mw
                + c4 * ln_PGV
            )

        elif self.kind == "near_fault_D50_gm":
            # Equation 7
            ln_PGV = np.log(self.PGV)

            if self.PGV <= 150:
                if self.Ts >= 0.1:
                    c1 = -7.718
                    c2 = 1.031
                    c3 = -0.48
                    c4 = 1.458
                elif self.Ts < 0.1:
                    c1 = -7.497
                    c2 = -2.731
                    c3 = 0.0
                    c4 = 1.458
            elif self.PGV > 150:
                if self.Ts >= 0.1:
                    c1 = -0.369
                    c2 = 1.031
                    c3 = -0.48
                    c4 = 0.025
                elif self.Ts < 0.1:
                    c1 = -0.148
                    c2 = -2.731
                    c3 = 0.0
                    c4 = 0.025

            ln_D = (
                c1
                - 2.931 * ln_ky
                - 0.319 * ln_ky**2
                + 0.584 * ln_ky * ln_Sa
                + 2.261 * ln_Sa
                - 0.241 * ln_Sa**2
                + c2 * self.Ts
                + c3 * (self.Ts) ** 2
                + 0.05 * self.Mw
                + c4 * ln_PGV
            )

        elif self.kind == "all_gm":
            # Equation 9
            ln_PGV = np.log(self.PGV)

            if self.PGV <= 115:
                if self.Ts >= 0.1:
                    a1 = -5.894
                    a2 = 3.152
                    a3 = -0.91
                    a4 = 0.0
                    a5 = 0.0
                elif self.Ts < 0.1:
                    a1 = -4.551
                    a2 = -9.69
                    a3 = 0.0
                    a4 = 0.0
                    a5 = 0.0
            elif self.PGV > 115:
                if self.Ts >= 0.1:
                    a1 = -5.894
                    a2 = 3.152
                    a3 = -0.91
                    a4 = 1.0
                    a5 = -4.75
                elif self.Ts < 0.1:
                    a1 = -4.551
                    a2 = -9.69
                    a3 = 0.0
                    a4 = 1.0
                    a5 = -4.75

            ln_D = (
                a1
                - 2.491 * ln_ky
                - 0.245 * ln_ky**2
                + 0.344 * ln_ky * ln_Sa
                + 2.703 * ln_Sa
                - 0.089 * ln_Sa**2
                + a2 * self.Ts
                + a3 * (self.Ts) ** 2
                + 0.607 * self.Mw
                + a4 * ln_PGV
                + a5
            )

        D = np.exp(ln_D)
        self.D = D
        self.result["non-zero_seismic_displacement [cm]"] = np.round(D, 2)

    def calculate_displacement_with_probability(
        self,
        probability: list,
        threshold_displacement: float,
    ):
        ln_ky = np.log(self.ky)
        ln_Sa = np.log(self.Sa_13Ts)
        d_thres = threshold_displacement

        if self.result["non-zero_seismic_displacement [cm]"] is None:
            self.calculate_nonzero_seismic_displacement()

        # Equation 2 / 4 / 6 / 8
        if self.kind == "ordinary_gm":
            # Equation 2
            self.sigma = 0.72
            if self.Ts <= 0.7:
                P_D_0 = 1 - norm.cdf(
                    -2.48
                    - 2.97 * ln_ky
                    - 0.12 * ln_ky**2
                    - 0.72 * self.Ts * ln_ky
                    + 1.70 * self.Ts
                    + 2.78 * ln_Sa
                )
            elif self.Ts > 0.7:
                P_D_0 = 1 - norm.cdf(
                    -3.42
                    - 4.93 * ln_ky
                    - 0.30 * ln_ky**2
                    - 0.35 * self.Ts * ln_ky
                    - 0.62 * self.Ts
                    + 2.86 * ln_Sa
                )

        elif self.kind == "near_fault_D100_gm":
            # Equation 4
            self.sigma = 0.56
            ln_PGV = np.log(self.PGV)

            if self.Ts <= 0.7:
                P_D_0 = (
                    1
                    + np.exp(
                        -10.787
                        - 8.717 * ln_ky
                        + 1.66 * ln_PGV
                        + 3.15 * self.Ts
                        + 7.56 * ln_Sa
                    )
                ) ** (-1)
            elif self.Ts > 0.7:
                P_D_0 = (
                    1
                    + np.exp(
                        -12.771
                        - 9.979 * ln_ky
                        + 2.286 * ln_PGV
                        - 4.965 * self.Ts
                        + 4.817 * ln_Sa
                    )
                ) ** (-1)

        elif self.kind == "near_fault_D50_gm":
            # Equation 6
            self.sigma = 0.54
            ln_PGV = np.log(self.PGV)

            if self.Ts <= 0.7:
                P_D_0 = (
                    1
                    + np.exp(
                        -14.930
                        - 10.383 * ln_ky
                        + 1.971 * ln_PGV
                        + 3.763 * self.Ts
                        + 8.812 * ln_Sa
                    )
                ) ** (-1)
            elif self.Ts > 0.7:
                P_D_0 = (
                    1
                    + np.exp(
                        -14.671
                        - 10.489 * ln_ky
                        + 2.222 * ln_PGV
                        - 4.759 * self.Ts
                        + 5.549 * ln_Sa
                    )
                ) ** (-1)

        elif self.kind == "all_gm":
            # Equation 8
            self.sigma = 0.736
            if self.Ts <= 0.7:
                P_D_0 = 1 - norm.cdf(
                    -2.46
                    - 2.98 * ln_ky
                    - 0.12 * ln_ky**2
                    - 0.71 * self.Ts * ln_ky
                    + 1.69 * self.Ts
                    + 2.76 * ln_Sa
                )
            elif self.Ts > 0.7:
                P_D_0 = 1 - norm.cdf(
                    -3.40
                    - 4.95 * ln_ky
                    - 0.30 * ln_ky**2
                    - 0.33 * self.Ts * ln_ky
                    - 0.62 * self.Ts
                    + 2.85 * ln_Sa
                )

        self.result["P(D='0')"] = np.round(P_D_0, 2)

        # equation 10 and 11
        P_D_d = (1 - P_D_0) * (
            norm.cdf((np.log(self.D) - np.log(d_thres)) / self.sigma)
        )
        self.result["P(D > d_threshold)"] = np.round(P_D_d, 2)

        list_displacement = []
        for prob in probability:
            if prob >= 1.0:
                raise ValueError("Probability should be in decimal!")
            elif prob < 0.0:
                raise ValueError("Probability should be positive!")

            try:
                d = prob_disp_bray(prob, P_D_0, self.D, self.sigma)
                list_displacement.append(d)
            except ValueError:
                list_displacement.append(np.nan)

        for i in range(len(list_displacement)):
            prob = probability[i]
            disp = np.round(list_displacement[i], 2)
            self.result["displacement_with_probability [cm]"][prob] = disp
