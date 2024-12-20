from functools import partial
from jax import numpy as jnp, jit


@jit
def planck(wavelength, temperature):
    h = 6.6260693e-34
    k = 1.3806503e-23
    c = 2.99792458e8
    
    c1 = 2 * h * c ** 2
    c2 = h * c / k
    val = c2 / ( wavelength * temperature )
    
    return c1 / ( wavelength ** 5 * jnp.expm1(val))


@jit
def flux_K_and_15um(temperature):
    # Given temperature, this returns the flux density at the K-band (coeffs calculated with PHOENIX models). Only valid from T = 2000-4300 K
    coeffsK = jnp.array([2.00415064e-09, -4.03013467e-05, 2.95556473e-01, -9.44430075e+02, 1.47617635e+06, -8.87817344e+08])
    coeffs15 = jnp.array([-2.36697609e-11, 3.92869705e-07, -2.59084993e-03, 8.46111093e+00, -1.34856686e+04, 8.51515082e+06])

    poly_K = jnp.polyval(coeffsK, temperature)
    poly_15 = jnp.polyval(coeffs15, temperature)
    
    return poly_K, poly_15


@jit
def thermal_emission_ratio(wavelength_um, rp_rs, T_p, T_s):
    wavelength = wavelength_um * 1e-6
    Bp = planck(wavelength, T_p)
    Bs = planck(wavelength, T_s)
    return 1e6 * rp_rs**2 * Bp / Bs # [ppm]


@jit
def Tday(Ts, aRs, AB, epsilon):
    # Cowan & Agol equation 4
    T0 = Ts * aRs ** -0.5
    albedo_factor = (1 - AB) ** 0.25
    efficiency_factor = (2/3 - 5/12 * epsilon) ** 0.25
    return T0 * albedo_factor * efficiency_factor


@jit
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


@jit
def target_cost(teff, aRs, AB_min, AB_max, eps_max, rp_rs, K_mag, n_sigma, eclipse_dur, one_eclipse_precision_hdl, photon_noise_excess):
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
    
    eclipses_for_n_sigma = jnp.ceil(
        (n_sigma * single_eclipse_error / 
         (fp_fs_no_redist - fp_fs_full_redist)) ** 2
    )
    cost_hours = jnp.array(
        eclipses_for_n_sigma * 
        (2 * eclipse_dur + 1.5) * 1.2
    )
    sort_order = jnp.argsort(cost_hours)
    
    return cost_hours, sort_order

