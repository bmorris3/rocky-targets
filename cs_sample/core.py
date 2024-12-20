import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import BlackBody
import astropy.units as u
from astropy.table import Table

__all__ = ['sheet']



sheet = None

max_program_limit = 500 # [hrs]


def download_sheet():
    global sheet
    url = (
        'https://docs.google.com/spreadsheets/d/'
        '1JJQvXVXUn2KLM8f9y5DmCEZbLVtzdhZm6bkU9Of27kg'
        '/gviz/tq?gid=2101251706&tqx=out:csv'
    )

    if sheet is None:
        sheet = pd.read_csv(url, header=0)[1:]
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
    x = np.linspace(3, 35, 1000)
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
    target_wavelength = 15 # µm
    ref_mag_wavelength = 2.19  # K band
    ref_sigma_eclipse = 94 * 4 ** 0.5
    ref_dur = 0.7005 # hrs
    ref_mag = 10.3 # K-band
    ref_teff = 2566 # K

    flux_K_teff, flux_15um_teff = flux_K_and_15um(teff)
    flux_K_ref, flux_15um_ref = flux_K_and_15um(ref_teff)
    ratio_at_ref_mag = flux_K_teff / flux_K_ref
    ratio_at_target_mag = flux_15um_teff / flux_15um_ref

    # Correct flux ratio to account for scaling at different magnitudes:
    flux_ratio = (
        10 ** (-0.3984 * (magnitude - ref_mag)) * 
        (ratio_at_target_mag / ratio_at_ref_mag)
    )

    return ( ref_sigma_eclipse / (duration / ref_dur) ** 0.5 ) / flux_ratio ** 0.5

def flux_K_and_15um(temperature):
    # Given temperature, this returns the flux density at the K-band (coeffs calculated with PHOENIX models). Only valid from T = 2000-4300 K
    coeffsK = np.array([2.00415064e-09, -4.03013467e-05, 2.95556473e-01, -9.44430075e+02, 1.47617635e+06, -8.87817344e+08])
    coeffs15 = np.array([-2.36697609e-11, 3.92869705e-07, -2.59084993e-03, 8.46111093e+00, -1.34856686e+04, 8.51515082e+06])

    return [np.polyval(coeffs, temperature)
            for coeffs in [coeffsK, coeffs15]]


def thermal_emission_ratio(wavelength, rp_rs, T_p, T_s):
    """planetStarEmission"""
    Bp = planck(wavelength, T_p)
    Bs = planck(wavelength, T_s)
    return 1e6 * rp_rs**2 * Bp / Bs # [ppm]


def Tday(Ts, aRs, AB, epsilon):
    # Cowan & Agol equation 4
    T0 = Ts * aRs ** -0.5
    albedo_factor = (1 - AB) ** 0.25
    efficiency_factor = (2/3 - 5/12 * epsilon) ** 0.25
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
