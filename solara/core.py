import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import BlackBody
import astropy.units as u
from astropy.table import Table
from urllib.request import urlretrieve

__all__ = ['sheet']

sheet = None

# parse reference solar system albedos for docs:
planets = pd.read_table(
    "solar_system.md",
    skiprows=[0, 1, 2, 3, 5],
    delimiter='|',
    usecols=[1, 2, 3],
    nrows=5
)
moons = pd.read_table(
    "solar_system.md",
    skiprows=list(range(17)) + [18],
    delimiter='|',
    usecols=[1, 2, 3],
    nrows=13
)

moons['A_B_value'] = [
    float(row) if '±' not in row
    else float(row.split('±')[0])
    for row in moons[moons.columns[1]]
]

albedo = 0.4

planet_columns = planets.columns
moon_columns = moons.columns

def download_sheet():
    global sheet
    url = (
        'https://docs.google.com/spreadsheets/d/'
        '1JJQvXVXUn2KLM8f9y5DmCEZbLVtzdhZm6bkU9Of27kg'
        # '/gviz/tq?gid=2101251706&tqx=out:csv'
        '/gviz/tq?gid=249855819&tqx=out:csv'
    )
    local_path = 'file.csv'
    if not os.path.exists(local_path):
        urlretrieve(url, filename=local_path)

    if sheet is None:
        # sheet = pd.read_csv(url, header=0)[1:]
        sheet = pd.read_csv(local_path)[1:]
        sheet['Rp'] = sheet[sheet.columns[1]]

    return sheet


def cosmic_shoreline_parameters():
    path = os.path.join(os.path.dirname(__file__), 'cs_nestor.txt')
    with open(path, 'r') as relation:
        cs = np.array([
            list(map(float, line.split()))
            for line in relation.read().splitlines()[1:]
        ])

    x, y = cs.T

    X = np.vander(np.log10(x), 2)

    slope, intercept = np.linalg.lstsq(X, np.log10(y), rcond=None)[0]
    return slope, intercept


def priority_from_cs_distance(v_esc, instellation):
    slope, intercept = cosmic_shoreline_parameters()
    x = np.linspace(5, 25, 1000)
    y = 10 ** (slope * np.log10(x) + intercept)

    # returns +1 for planets on the "retains atmosphere" side of the shoreline,
    # returns -1 for planets on the "loses atmosphere" side of the shoreline:
    retains_or_loses = 2 * (
            10 ** (slope * np.log10(v_esc) + intercept) > instellation
    ).astype(int) - 1

    priority = np.hypot(
        np.log10(v_esc)[:, None] - np.log10(x)[None, :],
        np.log10(instellation)[:, None] - np.log10(y)[None, :]
    ).min(axis=1) * retains_or_loses

    return priority, x, y


def eclipse_error(flux_ratio, rp_rs, duration, magnitude, teff):
    target_wavelength = 15  # µm
    ref_mag_wavelength = 2.19  # K band
    ref_sigma_eclipse = 94 * 4 ** 0.5
    ref_dur = 0.7005  # hrs
    ref_mag = 10.3  # K-band
    ref_teff = 2566  # K

    flux_K_teff, flux_15um_teff = flux_K_and_15um(teff)
    flux_K_ref, flux_15um_ref = flux_K_and_15um(ref_teff)
    ratio_at_ref_mag = flux_K_teff / flux_K_ref
    ratio_at_target_mag = flux_15um_teff / flux_15um_ref

    # Correct flux ratio to account for scaling at different magnitudes:
    flux_ratio = (
            10 ** (-0.3984 * (magnitude - ref_mag)) *
            (ratio_at_target_mag / ratio_at_ref_mag)
    )

    return (ref_sigma_eclipse / (duration / ref_dur) ** 0.5) / flux_ratio ** 0.5


def flux_K_and_15um(temperature):
    # Given temperature, this returns the flux density at the K-band (coeffs calculated with PHOENIX models). Only valid from T = 2000-4300 K
    coeffsK = np.array(
        [2.00415064e-09, -4.03013467e-05, 2.95556473e-01, -9.44430075e+02, 1.47617635e+06, -8.87817344e+08])
    coeffs15 = np.array(
        [-2.36697609e-11, 3.92869705e-07, -2.59084993e-03, 8.46111093e+00, -1.34856686e+04, 8.51515082e+06])

    return [np.polyval(coeffs, temperature)
            for coeffs in [coeffsK, coeffs15]]


def thermal_emission_ratio(wavelength, rp_rs, T_p, T_s):
    """planetStarEmission"""
    Bp = planck(wavelength, T_p)
    Bs = planck(wavelength, T_s)
    return 1e6 * rp_rs ** 2 * Bp / Bs  # [ppm]


def Tday(Ts, aRs, AB, epsilon):
    # Cowan & Agol equation 4
    T0 = Ts * aRs ** -0.5
    albedo_factor = (1 - AB) ** 0.25
    efficiency_factor = (2 / 3 - 5 / 12 * epsilon) ** 0.25
    return T0 * albedo_factor * efficiency_factor


def parse_dang2024():
    dang2024 = Table.read('cs_sample/dang2024_table3.txt', format='ascii.latex')

    for col in dang2024.colnames[1:]:
        for newcol in [col + '_mean', col + '_err']:
            if newcol.endswith('mean'):
                dang2024[newcol] = np.array([row.split('^')[0][1:] for row in dang2024[col]]).astype(float)
            else:
                errs = [row.split('^')[1].split('_') for row in dang2024[col]]
                errs = [[abs(float(e.strip().split('}')[0][1:])) for e in row] for row in errs]
                dang2024[newcol] = np.mean(errs, axis=1).astype(float)

    # fig, ax = plt.subplots()
    # ax.errorbar(
    #     dang2024['AB_mean'], dang2024['epsilon_mean'],
    #     xerr=dang2024['AB_err'], yerr=dang2024['epsilon_err'], fmt='o'
    # )
    return dang2024


def planck(wavelength, temperature):
    h = 6.6260693e-34
    k = 1.3806503e-23
    c = 2.99792458e8

    c1 = 2 * h * c ** 2
    c2 = h * c / k
    val = c2 / (wavelength * temperature)

    return c1 / (wavelength ** 5 * np.expm1(val))


def flux_K_and_15um(temperature):
    # Given temperature, this returns the flux density at the K-band (coeffs calculated with PHOENIX models). Only valid from T = 2000-4300 K
    coeffsK = np.array(
        [2.00415064e-09, -4.03013467e-05, 2.95556473e-01, -9.44430075e+02, 1.47617635e+06, -8.87817344e+08])
    coeffs15 = np.array(
        [-2.36697609e-11, 3.92869705e-07, -2.59084993e-03, 8.46111093e+00, -1.34856686e+04, 8.51515082e+06])

    poly_K = np.polyval(coeffsK, temperature)
    poly_15 = np.polyval(coeffs15, temperature)

    return poly_K, poly_15


def thermal_emission_ratio(wavelength_um, rp_rs, T_p, T_s):
    wavelength = wavelength_um * 1e-6
    Bp = planck(wavelength, T_p)
    Bs = planck(wavelength, T_s)
    return 1e6 * rp_rs ** 2 * Bp / Bs  # [ppm]


def Tday(Ts, aRs, AB, epsilon):
    # Cowan & Agol equation 4
    T0 = Ts * aRs ** -0.5
    albedo_factor = (1 - AB) ** 0.25
    efficiency_factor = (2 / 3 - 5 / 12 * epsilon) ** 0.25
    return T0 * albedo_factor * efficiency_factor


def eclipse_error(flux_ratio, rp_rs, duration, magnitude, teff):
    target_wavelength = 15  # µm
    ref_mag_wavelength = 2.19  # K band
    ref_sigma_eclipse = 94 * 4 ** 0.5
    ref_dur = 0.7005  # hrs
    ref_mag = 10.3  # K-band
    ref_teff = 2566  # K

    flux_K_teff, flux_15um_teff = flux_K_and_15um(teff)
    flux_K_ref, flux_15um_ref = flux_K_and_15um(ref_teff)
    ratio_at_ref_mag = flux_K_teff / flux_K_ref
    ratio_at_target_mag = flux_15um_teff / flux_15um_ref

    # Correct flux ratio to account for scaling at different magnitudes:
    flux_ratio = (
            10 ** (-0.3984 * (magnitude - ref_mag)) *
            (ratio_at_target_mag / ratio_at_ref_mag)
    )

    return (ref_sigma_eclipse / (duration / ref_dur) ** 0.5) / flux_ratio ** 0.5


def target_cost(teff, aRs, AB_min, AB_max, eps_max, rp_rs, K_mag, n_sigma, eclipse_dur, one_eclipse_precision_hdl,
                photon_noise_excess, overhead_fraction=0.2):
    dayside_temperature_no_redist = Tday(
        teff, aRs,
        AB=AB_min,
        epsilon=0
    )
    dayside_temperature_full_redist = Tday(
        teff, aRs,
        AB=AB_max,
        epsilon=eps_max
    )
    fp_fs_no_redist = thermal_emission_ratio(
        15, rp_rs,
        dayside_temperature_no_redist,
        teff
    )
    fp_fs_full_redist = thermal_emission_ratio(
        15, rp_rs,
        dayside_temperature_full_redist,
        teff
    )
    scaled_single_eclipse_error = eclipse_error(
        fp_fs_no_redist,
        rp_rs,
        eclipse_dur,
        K_mag, teff
    )

    # allow scaling of eclipse depth precision between the one estimated from scaling TRAPPIST-1 c
    # and the photon-noise-limited eclipse depth precision from Pandeia.
    # photon_noise_excess==0 returns Pandeia precision, photon_noise_excess==1 returns
    # the scaled TRAPPIST-1 c precision, photon_noise_excess > 1 would represent worse noise than T-1 c.
    single_eclipse_error = (
            one_eclipse_precision_hdl +
            photon_noise_excess * (scaled_single_eclipse_error - one_eclipse_precision_hdl)
    )

    eclipses_for_n_sigma = np.ceil(
        (n_sigma * single_eclipse_error /
         (fp_fs_no_redist - fp_fs_full_redist)) ** 2
    )
    cost_hours = np.array(
        eclipses_for_n_sigma *
        (2 * eclipse_dur + 1.5) * (1 + overhead_fraction)
    )
    sort_order = np.argsort(cost_hours)

    return cost_hours, sort_order


def closest_albedos(albedo):
    min_planet_ind = np.argmin(np.abs(albedo - planets[planet_columns[1]]))
    min_moon_ind = np.argmin(np.abs(albedo - moons['A_B_value']))

    def strip_or_pass(x):
        if hasattr(x, 'strip'):
            return x.strip()
        return x

    planet_name, planet_A_B, planet_ref = map(strip_or_pass, list(planets.iloc[min_planet_ind]))
    moon_name, moon_A_B, moon_ref, _ = map(strip_or_pass, list(moons.iloc[min_moon_ind]))
    return (
        (planet_name, planet_A_B, planet_ref),
        (moon_name, moon_A_B, moon_ref)
    )