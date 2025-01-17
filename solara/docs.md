# 1. Targets under consideration (TUC)

Review the targets under consideration (TUC) in the public document on
the
[STScI Outerspace page](https://outerspace.stsci.edu/pages/viewpage.action?pageId=257035126).
The list contains 80 planets.

# 2. Eclipse depth precision

Published photometry from JWST/MIRI at 15 µm has slightly greater than
the precision predicted from photon noise estimates. In this tool, we
provide eclipse precision estimates by 1) assuming photon noise-limited
photometry; and 2) scaling the measured eclipse precision of TRAPPIST-1 c.

### 2a. Photon noise-limited precision estimates with Pandeia

Hannah Diamond-Lowe ran simulated eclipse observations with Pandeia to compute the single
eclipse depth precision, using stellar spectra from the PHOENIX grid and choosing the
subarray mode for each target.

### 2b. Scaled precision estimates from TRAPPIST-1 c observations

Néstor Espinoza estimated the eclipse precision for each target by scaling the
eclipse precision measured by
[Zieba et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023Natur.620..746Z/abstract)
for TRAPPIST-1 c of 94 ppm after four eclipses. We scale the TRAPPIST-1 c precision
to the precision for another star-planet system by scaling for the target host
star's magnitude relative to TRAPPIST-1, and scaling for the relative durations
of the target's eclipse and TRAPPIST-1 c's eclipse duration.

### 2c. One, the other, or somewhere in between

In the ``OBS`` tab, there is a slider called ``noise excess``. When set to zero, the
cost calculation uses the photon noise-limited precision. When set to one, the
cost calculation uses the scaled precision from the TRAPPIST-1 c observations. When
set to other values, it scales linearly between or beyond those two cases. Setting
``noise excess > 1`` represents the assumption that the typical noise is usually
greater than Zieba et al. observed for TRAPPIST-1 c.

# 3. Cost: required duration of observations

We estimate the number of hours required to rule out an atmosphere
by comparing the expected eclipse depth precision against the difference
between the expected eclipse depth *with* and *without* an atmosphere
(minimum vs. maximum eclipse depth at 15 µm).

### 3a. Planet-star flux ratio

We compute the ratio of the thermal emission from the planet and star by
assuming blackbody emission at 15 µm,

$$ \delta_{\rm eclipse} = \frac{F_p}{F_\star} = \left(\frac{R_p}{R_\star}\right)^2 \frac{B_\lambda(T_{\rm day})}{B_\lambda(T_\star)}.$$

The dayside temperature $T_{\rm day}$ of a planet is (e.g. [Cowan & Agol, 2011](https://ui.adsabs.harvard.edu/abs/2011ApJ...729...54C/abstract)):

$$ T_{\rm day} = T_{\star} \left(\frac{R_{\star}}{a}\right)^{1/2} \left(1 - A_{\rm B}\right)^{1/4} \left(\frac{2}{3} - \frac{5}{12} \epsilon\right)^{1/4},$$

where $A_{\rm B}$ is the Bond albedo, and $\epsilon$ is the heat redistribution efficiency.

### 3b. Bond albedo

The Bond albedo is the ratio of the power scattered by a planet's atmosphere and
the total power radiated into the planet. Airless bodies in the solar system have
small Bond albedos, such as $A_{\rm B,~Moon} = 0.1$ and $A_{\rm B,~Mercury} = 0.08$.
Thick atmospheres have higher albedos. Earth, Neptune, and Venus have
 $A_{\rm B} = \\{0.3, 0.3, 0.7\\}$ respectively. You can set the
Bond albedo of the planet in cases with and without an atmosphere from the ``SYSTEM`` tab.

### 3c. Heat redistribution

We expect the heat redistribution efficiency $\epsilon \rightarrow 0$ for planets without an
atmosphere. The cost calculations assume $\epsilon=0$ for the no-atmosphere case, and you can
adjust the heat redistribution of the with-an-atmosphere case is the ``SYSTEM`` tab with the
``Redist. efficiency`` slider. For reference, Venus and Earth have heat redistribution
efficiency near unity, while Mercury has heat redistribution near zero.

### 3d. Observation duration

We compute the observing cost from the eclipse depth with an atmosphere $\delta_{\rm min}$ and
the depth without an atmosphere $\delta_{\rm max}$. We compute the planet-star emission ratio $F_p/F_\star$
for the planet with and without an atmosphere via $T_{\rm day}(A_{\rm B}, \epsilon)$ (see Section 3a).
We compute the uncertainty on the depth of a single eclipse observation as in Section 2.

We set the SNR requirement in the ``OBS`` tab with the slider labeled ``Require detection above N sigma``.
The minimum number of eclipse observations required to meet or exceed ``N sigma`` significance is

$$ N_{\rm eclipses, min} = {\rm ceil}\left( \frac{N_{\sigma, \rm required} ~~ \sigma_{\delta}}{\delta_{\rm max} - \delta_{\rm min}} \right)^2.$$

The cost in hours is

$$ {\rm cost} = N_{\rm eclipses, min} (2 \Delta T + 1.5 {\rm ~hours}) (1 + f), $$

where we assume observations span at least twice the eclipse duration $\Delta T$. We additional
a minimum of 1.5 hours for acquisition and settling, and an additional overhead fraction $f$.
Typical MIRI visits incur overheads between 15-30%. You can adjust this parameter in the ``OBS`` tab
 with the ``Fractional overhead`` slider.

# 4. Results

### 4a. Figures

The plots in the top left shows the targets which meet the selection criteria in the app,
as well the full Targets Under Consideration list. The row of histograms at the top show
the host star effective temperature, the required observation duration per target, the
"priority" metric (log-distance from the theoretical cosmic shoreline), the bulk planetary
density, and log10 XUV instellation. The gray histograms show all targets in the TUC,
blue histograms show the "selected" targets in the "TARGETS" table beneath the plot.

The big plot shows the cosmic shoreline from
[Zahnle & Catling (2017)](https://ui.adsabs.harvard.edu/abs/2017ApJ...843..122Z/abstract).
Targets that do not meet the selection criteria are plotted with gray markers. Scheduled targets
are shown with colored markers, and targets that meet the criteria but would not fit within the maximum
program duration are shown as unfilled black circles. The size of the marker is proportional to the
observation cost, and color indicates log distance from the cosmic shoreline reference line shown in gray.

You can choose the vertical axis of the cosmic shoreline plot – either the full-spectrum instellation *or* the
XUV instellation – by toggling a checkbox in the ``INCLUDE`` tab.

### 4b. Targets

The table shown on the left in the ``TARGETS`` tab shows all required targets (selected in the ``INCLUDE`` tab),
plus the next $N$ lowest-cost targets that fit within the maximum program length, which can be adjusted in the
``OBS`` tab with the ``Max total obs. hours`` slider. The first rows will be the required targets, followed
by targets that cost no more than the maximum duration, sorted in order of cost.

Targets are shown in the cosmic shoreline plot on the left with colored markers. The markers in the plot
are labeled with indices that correspond to the index in the "marker" column of the table below.

### 4c. Unscheduled

The table shown on the left in the ``UNSCHEDULED`` tab shows targets that meet the constraints
set within the app that *do not* fit within the maximum program duration, sorted in cost order.

Unscheduled planets are shown in the cosmic shoreline plot on the left with unfilled black circles.



