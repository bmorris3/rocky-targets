import os
from packaging.version import Version
import solara
import numpy as np
import markdown

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from IPython.display import display

import astropy.units as u
from astropy import __version__ as astropy_version
from astropy.table import Table
from astropy.visualization import simple_norm
from ipywidgets import (
    interactive, FloatSlider, HBox, VBox, Button,
    Layout, Box, Checkbox, Output, IntSlider,
    SelectMultiple, Label, HTML
)

from core import (
    download_sheet, priority_from_cs_distance, target_cost,
    closest_albedos, density_of_earth_composition
)

sheet = download_sheet()


@solara.component
def Page():
    columns = [
        'Teff', 'Kmag', 'Rp/Rs',
        'a/Rs', 'Eclipse Dur',
        'Instellation', 'Escape Velocity', 'Has < 20% mass constraint?',
        'Rp', 'Mp', 'XUV Instellation',
        'Teq', '1 eclipse depth precision',
        'Mp_err'
    ]

    (
        teff, kmag, rp_rs,
        aRs, eclipse_dur,
        instellation, v_esc, mass_constraint,
        Rp, Mp, xuv,
        Teq, one_eclipse_precision_hdl,
        Mp_err
    ) = np.array(
        sheet[columns].to_numpy().T
    )

    names = np.array([t.split('(')[0].strip() for t in sheet['Planet name']])
    mass_precision = Mp_err / Mp

    priority, x, y = priority_from_cs_distance(v_esc, instellation)

    planet_mass = Mp * u.M_earth
    rho_earth = density_of_earth_composition(Mp) * u.g / u.cm ** 3
    density = (
            (planet_mass / (4 / 3 * np.pi * (Rp * u.R_earth) ** 3)) / rho_earth
    ).to_value(u.dimensionless_unscaled)

    in_go_programs = np.isin(
        sheet['Planet name'].tolist(),
        ['LP 791-18 d', 'TRAPPIST-1 b', 'TRAPPIST-1 c']
    )

    in_hot_rocks = np.array([
        "Hot Rocks" in comment if isinstance(comment, str) else False
        for comment in sheet['General comments'].tolist()
    ])

    norm = simple_norm(priority, 'linear', min_cut=-0.1, max_cut=priority.max())

    scenarios = dict(
        TUC_v1=dict(
            eps_max=1,
            AB_min=0.1,
            AB_max=0.3,
            n_sigma=4,
            teff_min=2500,
            teff_max=4000,
            require_targets=['GJ 3929 b', 'LTT 1445 A c'],
            exclude_targets=[]
        ),

        mercury_vs_venus=dict(
            eps_max=1.0,  # perfect redist
            AB_min=0.119,  # Mercury
            AB_max=0.75,  # Venus
            n_sigma=3
        ),

        mercury_vs_earth=dict(
            eps_max=1.0,  # perfect redist
            AB_min=0.119,  # Mercury
            AB_max=0.29,  # Earth
            n_sigma=3
        ),

        mercury_vs_mars=dict(
            eps_max=0.04,  # Mars
            AB_min=0.119,  # Mercury
            AB_max=0.16,  # Mars
            n_sigma=3
        ),
    )

    var_to_markdown = {
        'eps_max': r'$\epsilon_{\rm max}$',
        'AB_min': r'$A_{\rm B, min}$',
        'AB_max': r'$A_{\rm B, max}$',
        'n_sigma': r'$N_\sigma$',
        'teff_min': r'$T_{\rm eff, min}$',
        'teff_max': r'$T_{\rm eff, max}$',
        'require_targets': 'Required targets',
        'exclude_targets': 'Excluded targets',
    }

    names_dict = {
        name: idx for name, idx in
        sorted(
            [[name, i]
             for i, name in
             enumerate(names)],
            key=lambda x: x[0]
        )
    }

    table_output = Output()
    fig_output = Output()

    plt.rcParams.update({'font.size': 10})

    eps_max, set_eps_max = solara.use_state(1)
    AB_min, set_AB_min = solara.use_state(0.1)
    AB_max, set_AB_max = solara.use_state(0.3)
    n_sigma, set_n_sigma = solara.use_state(4)
    teff_min, set_teff_min = solara.use_state(3200)
    teff_max, set_teff_max = solara.use_state(3800)
    teq_max, set_teq_max = solara.use_state(600)
    max_hrs, set_max_hrs = solara.use_state(500)
    noise_excess, set_noise_excess = solara.use_state(1)
    overhead_fraction, set_overhead_fraction = solara.use_state(0.2)
    mass_prec, set_mass_prec = solara.use_state(0.2)
    include_go, set_include_go = solara.use_state(False)
    include_hot_rocks, set_include_hot_rocks = solara.use_state(False)
    include_imprecise_mass, set_include_imprecise_mass = solara.use_state(False)
    use_xuv, set_use_xuv = solara.use_state(False)
    require_targets, set_require_targets = solara.use_state(['GJ 3929 b', 'LTT 1445 A c'])
    exclude_targets, set_exclude_targets = solara.use_state([])

    with solara.Columns():
        with solara.Column(align='center'):
            with solara.Row():
                require_targets_idx = [names_dict[target_name] for target_name in require_targets]
                exclude_targets_idx = [names_dict[target_name] for target_name in exclude_targets]

                required_mask = np.isin(np.arange(len(teff)), np.array(require_targets_idx))
                exclude_mask = np.isin(np.arange(len(teff)), np.array(exclude_targets_idx))
                mask = required_mask | exclude_mask

                # exclude outside of temperature range
                mask |= ~np.array((teff_min < teff) & (teff < teff_max))

                # apply max T_eq cutoff
                mask |= ~(Teq < teq_max)

                if not include_go:
                    mask |= in_go_programs

                if not include_hot_rocks:
                    mask |= in_hot_rocks

                if not include_imprecise_mass:
                    mask |= ((mass_precision > mass_prec) | (mass_precision == 0))

                cost, sort_order, eclipses_for_n_sigma = target_cost(
                    teff=teff,
                    aRs=aRs,
                    AB_min=AB_min,
                    AB_max=AB_max,
                    eps_max=eps_max,
                    rp_rs=rp_rs,
                    K_mag=kmag,
                    n_sigma=n_sigma,
                    eclipse_dur=eclipse_dur,
                    one_eclipse_precision_hdl=one_eclipse_precision_hdl,
                    photon_noise_excess=noise_excess,
                    overhead_fraction=overhead_fraction
                )

                cost_required = np.sum(cost[required_mask])

                sort = np.argsort(cost[~(mask | required_mask)])
                last_index = np.searchsorted(
                    np.cumsum(cost[~(mask | required_mask)][sort]),
                    max_hrs - cost_required
                )
                acceptable_indices = np.arange(len(cost))[
                     ~(mask | required_mask)
                 ]
                unacceptable_indices = np.arange(len(cost))[
                     (mask | required_mask)
                 ]
                sheet_mask = acceptable_indices[sort][:last_index]
                sheet_mask = np.concatenate(
                    [require_targets_idx, sheet_mask]
                ).astype(int)
                sheet_mask_excluded = acceptable_indices[~np.isin(acceptable_indices, sheet_mask)]

                with fig_output:
                    fig_output.clear_output()

                    fig = plt.figure(figsize=(7, 6), dpi=250)
                    gs = GridSpec(5, 5, figure=fig)

                    ax_hist = [fig.add_subplot(gs[0, i]) for i in range(gs.ncols)]
                    ax = fig.add_subplot(gs[1:, :])
                    if use_xuv:
                        # from Zahnle & Catling 2017
                        yi = 1e-6 / (0.18 ** 4) * x ** 4
                        ax.loglog(x, yi, lw=3, color='silver', zorder=-100, alpha=0.5)
                    else:
                        ax.loglog(x, y, lw=3, color='silver', zorder=-100, alpha=0.5)

                    plot_instell = xuv if use_xuv else instellation
                    ax.scatter(
                        v_esc[unacceptable_indices], plot_instell[unacceptable_indices],
                        edgecolor='none',
                        color='silver',
                        alpha=0.3
                    )
                    c1 = 1e-4
                    c2 = 3
                    c3 = 2.5
                    c4 = 10
                    cax = ax.scatter(
                        v_esc[sheet_mask],
                        plot_instell[sheet_mask],
                        c=priority[sheet_mask],
                        edgecolor='none',
                        norm=norm,
                        s=c1 * (c2 * cost[sheet_mask]) ** c3 + c4,
                    )
                    ax.scatter(
                        v_esc[sheet_mask_excluded],
                        plot_instell[sheet_mask_excluded],
                        # c=priority[sheet_mask_excluded],
                        facecolors='none',
                        edgecolors='k',
                        norm=norm,
                        s=c1 * (c2 * cost[sheet_mask_excluded]) ** c3 + c4,
                    )
                    plt.colorbar(cax, ax=ax, label='priority', fraction=0.08)
                    if sum(sheet_mask):
                        handles, labels = cax.legend_elements(
                            prop="sizes", alpha=0.6, mec='none',
                            num=4, fmt='{x:.0f}', func=lambda s: ((s - c4) / c1) ** (1/c3) / c2
                        )
                        ax.legend(handles, labels, loc="lower left", title="Cost [hr]")

                    for i, (xi, yi) in enumerate(zip(v_esc[sheet_mask], plot_instell[sheet_mask])):
                        ax.annotate(f' {i}', (xi, yi), ha='left', va='bottom', fontsize=9)

                    xticks = [6, 10, 20]
                    ax.set(
                        xlabel='$v_{\\rm esc}$ [km s$^{-1}$]',
                        ylabel=('XUV ' if use_xuv else '') + 'Instellation [I$_{\oplus}$]',
                        xscale='log',
                        yscale='log',
                    )
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticks)

                    tables = []
                    all_indices = np.arange(len(cost))
                    req_excl = np.array(len(cost) * [''])
                    require_check = np.isin(all_indices, require_targets_idx)
                    req_excl[require_check] = '✅'
                    for flippable_mask in [sheet_mask, sheet_mask_excluded]:
                        table_contents = {
                            'target': names[flippable_mask],
                            'required': req_excl[flippable_mask],
                            'cost [hr]': cost[flippable_mask],
                            'eclipses': eclipses_for_n_sigma[flippable_mask].astype(int),
                            'priority': priority[flippable_mask],
                            'mass unc.': mass_precision[flippable_mask],
                            '$\\rho$ [$\\rho_\\oplus$]': density[flippable_mask],
                            '$T_{\\rm eq}$ [K]': Teq[flippable_mask],
                            '$v_{\\rm esc}$ [km/s]': v_esc[flippable_mask],
                        }

                        if use_xuv:
                            table_contents['$I_{\\rm XUV}$ [$I_\oplus$]'] = xuv[flippable_mask]
                        else:
                            table_contents['I [$I_\oplus$]'] = instellation[flippable_mask]

                        mask_cols = ['GO', 'HotRocks', 'ImpMass']
                        for toggle, mask, hdr in zip(
                                [include_go, include_hot_rocks, include_imprecise_mass],
                                [in_go_programs, in_hot_rocks, ~mass_constraint.astype(bool)],
                                mask_cols
                        ):
                            if toggle:
                                table_contents[hdr] = np.where(mask[flippable_mask], '❌', '')

                        tbl = Table(table_contents)
                        for col in tbl.colnames[1:]:
                            if col not in mask_cols and col not in ['required']:
                                tbl[col].format = '0.1f'
                            if col in ['eclipses']:
                                tbl[col].format = '1d'
                            if col in ['mass unc.']:
                                tbl[col].format = '0.2f'

                        tables.append(tbl)

                    target_table, excluded_target_table = tables
                    excluded_target_table.sort('cost [hr]')
                    notes = (
                            f'N$_{{\\rm targets}}$ = {len(sheet_mask)}\n' +
                            f'Total obs time = {cost[sheet_mask].sum():.0f} hrs\n'
                    )
                    ax.annotate(
                        notes, (0.05, 0.95),
                        xycoords='axes fraction',
                        va='top', ha='left',
                        fontsize=12
                    )

                    labels = ['$T_{\\rm eff}$ [K]', '$t_{\\rm obs}$ [hrs]', 'priority',
                              '$\\rho$ [$\\rho_\\oplus$]', 'log XUV']

                    for i, (parameter, label) in enumerate(zip(
                            [teff, cost, priority, density, np.where(xuv < 1e4, np.log10(xuv), np.nan)],
                            labels
                    )):
                        if i not in [1]:
                            n, bins = ax_hist[i].hist(
                                parameter, alpha=0.2, color='silver'
                            )[:2]
                        else:
                            bins = None

                        ax_hist[i].hist(
                            parameter[sheet_mask], color='C0', bins=bins
                        )
                        ax_hist[i].set(
                            xlabel=label
                        )
                        if any(s in label for s in ['T_{', 'priority', 'rho', 'XUV']):
                            ax_hist[i].set_yscale('log')

                    fig.tight_layout()
                    deps = [
                        eps_max,
                        AB_min,
                        AB_max,
                        n_sigma,
                        teff_min,
                        teff_max,
                        teq_max,
                        max_hrs,
                        noise_excess,
                        overhead_fraction,
                        mass_prec,
                        include_go,
                        include_hot_rocks,
                        include_imprecise_mass,
                        use_xuv,
                        require_targets,
                        exclude_targets
                    ]

                    solara.FigureMatplotlib(fig, dependencies=deps)
            with solara.Row():
                with solara.lab.Tabs(background_color="primary", dark=True):
                        with solara.lab.Tab("Targets"):
                            with solara.Column(align='center'):
                                with table_output:
                                    table_output.clear_output()
                                    show_in_notebook_kwargs = dict(display_length=-1, show_row_index='marker')
                                    if Version(astropy_version) >= Version('7.0'):
                                        show_in_notebook_kwargs['backend'] = 'classic'
                                    display(target_table.show_in_notebook(**show_in_notebook_kwargs))
                                solara.Markdown(
                                    '"**Targets**" are planets selected to provide the maximum number of targets '
                                    'that meet your selection criteria and fit within the maximum '
                                    f'program duration ({max_hrs} hrs). Each planet in the "Targets" list is plotted '
                                    'above with a filled and colored point, and labeled with the number in the '
                                    '"marker" column in the table. '
                                    '"**Unscheduled**" targets meet your selection criteria '
                                    'but do not fit within the maximum program duration. "Unscheduled" targets '
                                    'are plotted above with unfilled black circles.'
                                )
                        with solara.lab.Tab("Unscheduled"):
                            with solara.Column(align='center'):
                                with table_output:
                                    table_output.clear_output()
                                    show_in_notebook_kwargs = dict(display_length=-1, show_row_index=None)
                                    if Version(astropy_version) >= Version('7.0'):
                                        show_in_notebook_kwargs['backend'] = 'classic'
                                    display(excluded_target_table.show_in_notebook(**show_in_notebook_kwargs))
                                solara.Markdown(
                                    '"**Unscheduled**" targets meet your selection criteria '
                                    'but do not fit within the maximum program duration.'
                                )
        slider_kwargs = dict(tick_labels='end_points', thumb_label='always')
        with solara.Column(margin=5):
            with solara.Row():
                with solara.lab.Tabs(background_color="primary", dark=True):
                    with solara.lab.Tab("System"):
                        solara.Markdown("## Planet-star system requirements")

                        solara.Details(summary="Click to expand: reference solar system Bond albedos",
                                       children=[solara.Markdown(open("solar_system.md").read())],
                                       expand=False)

                        solara.Markdown("<br /><br />")
                        solara.Markdown("#### Planet: deepest eclipse")
                        solara.FloatSlider(
                            'Bond albedo, no redist.',
                            value=AB_min, on_value=set_AB_min, min=0, max=1, step=0.01,
                            **slider_kwargs
                        )
                        (
                            (planet_name, planet_A_B, planet_ref),
                            (moon_name, moon_A_B, moon_ref)
                        ) = closest_albedos(AB_min)
                        solara.Markdown(
                            f"Closest to {planet_name} with $A_{{\\rm B}} = {planet_A_B}$ "
                            f"({planet_ref}), or {moon_name} with $A_{{\\rm B}} = {moon_A_B}$ "
                            f"({moon_ref})."
                        )
                        solara.Markdown("<br /><br />")

                        solara.Markdown("#### Planet: shallowest eclipse")
                        solara.FloatSlider(
                            'Redist. efficiency',
                            value=eps_max, on_value=set_eps_max, min=0, max=1, step=0.01,
                            **slider_kwargs
                        )
                        solara.Markdown("<br /><br />")
                        solara.FloatSlider(
                            'Bond albedo, with redist.',
                            value=AB_max, on_value=set_AB_max, min=0, max=1, step=0.01,
                            **slider_kwargs
                        )
                        (
                            (planet_name, planet_A_B, planet_ref),
                            (moon_name, moon_A_B, moon_ref)
                        ) = closest_albedos(AB_max)
                        solara.Markdown(
                            f"Closest to {planet_name} with $A_{{\\rm B}} = {planet_A_B}$ "
                            f"({planet_ref}), or {moon_name} with $A_{{\\rm B}} = {moon_A_B}$ "
                            f"({moon_ref})."
                        )
                        solara.Markdown("<br /><br />")

                        solara.Markdown("#### Temperature")
                        solara.IntSlider(
                            'Minimum stellar T_eff',
                            value=teff_min, on_value=set_teff_min, min=2500, max=4000, step=10,
                            **slider_kwargs
                        )
                        solara.Markdown("<br /><br />")
                        solara.IntSlider(
                            'Maximum stellar T_eff',
                            value=teff_max, on_value=set_teff_max, min=2500, max=4000, step=10,
                            **slider_kwargs
                        )
                        solara.Markdown("<br /><br />")
                        solara.IntSlider(
                            'Max planet T_eq',
                            value=teq_max, on_value=set_teq_max, min=200, max=1000, step=10,
                            **slider_kwargs
                        )

                    with solara.lab.Tab("Obs"):
                        solara.Markdown("## Observation requirements")

                        solara.FloatSlider(
                            'Max frac mass precision',
                            value=mass_prec, min=0, max=0.5, step=0.01, on_value=set_mass_prec,
                            **slider_kwargs
                        )
                        solara.Markdown("<br /><br />")
                        solara.FloatSlider(
                            'Require detection above N sigma',
                            value=n_sigma, on_value=set_n_sigma, min=3, max=6, step=0.1,
                            **slider_kwargs
                        )
                        solara.Markdown("<br /><br />")
                        solara.IntSlider(
                            'Max total obs. hours',
                            value=max_hrs, on_value=set_max_hrs, min=500, max=700, step=10,
                            **slider_kwargs
                        )
                        solara.Markdown("<br /><br /><br /><br />")
                        solara.FloatSlider(
                            'noise excess',
                            value=noise_excess, on_value=set_noise_excess, min=0, max=3, step=0.1,
                            **slider_kwargs
                        )
                        solara.Markdown(
                            'The `noise excess` parameter scales the precision on the eclipse depth used '
                            'to calculate the required observing time for a given target. It ranges from '
                            'a photon-noise limited estimate calculated with Pandeia, corresponding to'
                            ' `noise excess == 0`, to the precision of the measured eclipse depth'
                            ' for TRAPPIST-1 c (scaled for planet\'s parameters), '
                            ' at `noise excess == 1`. `noise excess > 1` is useful for testing scenarios '
                            'where the noise exceeds the noise observed in TRAPPIST-1 c observations. '
                        )
                        solara.Markdown("<br /><br /><br /><br />")
                        solara.FloatSlider(
                            'Fractional overhead',
                            value=overhead_fraction, on_value=set_overhead_fraction, min=0.15, max=0.5, step=0.05,
                            **slider_kwargs
                        )
                        solara.Markdown(
                            'Typical visits with JWST/MIRI incur overheads which increase the hours charged '
                            'by 15-30% over the observing time.'
                        )

                    with solara.lab.Tab("Include"):
                        solara.Markdown("### Allow precluded targets")
                        solara.Markdown(
                            "See which precluded targets would fit within the program requirements "
                            "by toggling the checkboxes below. The precluded targets will have a \"❌\" in "
                            "the column corresponding to their preclusion criterion."
                        )
                        solara.Checkbox(
                            label='Include GO', value=include_go,
                            on_value=set_include_go
                        )
                        solara.Checkbox(
                            label='Include Hot Rocks',
                            value=include_hot_rocks,
                            on_value=set_include_hot_rocks,
                        )
                        solara.Checkbox(
                            label='Include imprecise masses',
                            value=include_imprecise_mass,
                            on_value=set_include_imprecise_mass,
                        )
                        solara.Markdown(
                            'Include targets with mass uncertainty >20%,'
                            'irrespective of the setting in the "Max frac mass precision" '
                            'slider in the "Obs" tab.'
                        )
                        solara.Markdown('<br /><br />')
                        solara.Markdown('### Specific targets')
                        solara.Markdown(
                            "**Note**: As of Jan 2025, the two pre-selected "
                            "targets are *GJ 3929 b* and *LTT 1445 A c*.<br /><br />"
                        )

                        solara.SelectMultiple(
                            'Require:', require_targets, sorted(list(names)), on_value=set_require_targets
                        )
                        solara.SelectMultiple(
                            'Exclude:', exclude_targets, sorted(list(names)), on_value=set_exclude_targets
                        )
                        solara.Markdown('<br /><br />')

                        solara.Markdown('### Plot options')
                        solara.Checkbox(
                            label='Show XUV instel.',
                            value=use_xuv, on_value=set_use_xuv
                        )

                    with solara.lab.Tab("Scenario"):
                        with solara.Column():
                            solara.Markdown("### Presets")
                            solara.Markdown("Select a combination of preset parameter values:")

                            sets = {
                                'eps_max': set_eps_max,
                                'AB_min': set_AB_min,
                                'AB_max': set_AB_max,
                                'n_sigma': set_n_sigma,
                                'teff_min': set_teff_min,
                                'teff_max': set_teff_max,
                                'teq_max': set_teq_max,
                                'max_hrs': set_max_hrs,
                                'noise_excess': set_noise_excess,
                                'overhead_fraction': set_overhead_fraction,
                                'mass_prec': set_mass_prec,
                                'include_go': set_include_go,
                                'include_hot_rocks': set_include_hot_rocks,
                                'include_imprecise_mass': set_include_imprecise_mass,
                                'use_xuv': set_use_xuv,
                                'require_targets': set_require_targets,
                                'exclude_targets': set_exclude_targets
                            }
                            for key, params in scenarios.items():

                                def handler(params=params):
                                    for param_name, param_value in params.items():
                                        sets[param_name](param_value)

                                solara.Button(f"{key.replace('_', ' ')}", on_click=handler)

                                description = []
                                for param_name, param_value in params.items():
                                    description.append(f"{var_to_markdown[param_name]} = {param_value}  ")

                                solara.Markdown(', '.join(description) + '<br /><br /><br />')

                    with solara.lab.Tab("Docs"):
                        with solara.Column():
                            solara.Markdown(open("docs.md").read())
