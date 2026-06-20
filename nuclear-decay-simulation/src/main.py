import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


# 1. time parameters

dt = 1.0  # 1 step = 1 day
time_steps = 2000

time_days = []

# 2. isotope data for the simplified nuclear chains

isotopes = {
    "Cs-137": {
        "half_life": 30.17 * 365.25,
        "daughter": "Ba-137m",
        "energy": 6.0
    },
    "Ba-137m": {
        "half_life": 2.6 / 24,
        "daughter": "Ba-137",
        "energy": 0.5
    },
    "Ba-137": {
        "half_life": None,
        "daughter": None,
        "energy": 0.0
    },
    "Sr-90": {
        "half_life": 28.8 * 365.25,
        "daughter": "Y-90",
        "energy": 5.0
    },
    "Y-90": {
        "half_life": 64 / 24,
        "daughter": "Zr-90",
        "energy": 7.0
    },
    "Zr-90": {
        "half_life": None,
        "daughter": None,
        "energy": 0.0
    }
}

# 3. initial start conditions

N = {
    "Cs-137": 1000,
    "Ba-137m": 0,
    "Ba-137": 0,
    "Sr-90": 500,
    "Y-90": 0,
    "Zr-90": 0
}

# 4. parameters for radiation shielding (simplified)

mu = 0.08
thickness = 10

# 5. storage arrays

Cs_vals = []
Sr_vals = []
Y_vals = []

radiation = []
radiation_shielded = []

# 6. simulation loop

for t in range(time_steps):

    current_time = t * dt
    time_days.append(current_time)

    new_N = N.copy()
    step_radiation = 0

    for iso, count in N.items():

        data = isotopes[iso]

        if data["half_life"] is None:
            continue

        lam = np.log(2) / data["half_life"]

        p_decay = 1 - np.exp(-lam * dt)

        decays = np.random.binomial(count, p_decay)

        daughter = data["daughter"]

        new_N[iso] -= decays
        new_N[daughter] += decays

        step_radiation += decays * data["energy"]

    N = new_N

    Cs_vals.append(N["Cs-137"])
    Sr_vals.append(N["Sr-90"])
    Y_vals.append(N["Y-90"])

    radiation.append(step_radiation)

    radiation_shielded.append(step_radiation * np.exp(-mu * thickness))

# 7. linear plots

plt.figure(figsize=(9,5))
plt.plot(time_days, Cs_vals, label="Cs-137")
plt.plot(time_days, Sr_vals, label="Sr-90")
plt.plot(time_days, Y_vals, label="Y-90")

plt.title("Isotope Populations Over Time (Linear Scale)")
plt.xlabel("Time (days)")
plt.ylabel("Atoms")
plt.grid(True)
plt.legend()
plt.savefig(
    "nuclear-decay-simulation/figures/isotope_decay_linear.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# 8. Isotope plot with log scale

plt.figure(figsize=(9,5))

plt.semilogy(time_days, Cs_vals, label="Cs-137")
plt.semilogy(time_days, Sr_vals, label="Sr-90")
plt.semilogy(time_days, Y_vals, label="Y-90")

plt.title("Isotope Populations Over Time (Log Scale)")
plt.xlabel("Time (days)")
plt.ylabel("Atoms (log scale)")
plt.legend()

plt.savefig(
    "nuclear-decay-simulation/figures/isotope_decay_log.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 9. radiation plots with log scale

plt.figure(figsize=(9,5))

plt.semilogy(time_days, radiation, label="Raw Radiation")
plt.semilogy(time_days, radiation_shielded, label="Shielded Radiation")

plt.title("Radiation Output (Log Scale)")
plt.xlabel("Time (days)")
plt.ylabel("Intensity (log scale)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.savefig(
    "nuclear-decay-simulation/figures/radiation_output.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

df = pd.DataFrame({
    "Time (days)": time_days,
    "Cs-137": Cs_vals,
    "Sr-90": Sr_vals,
    "Y-90": Y_vals,
    "Radiation": radiation,
    "Shielded Radiation": radiation_shielded
})
import os

df = pd.DataFrame({
    "Time (days)": time_days,
    "Cs-137": Cs_vals,
    "Sr-90": Sr_vals,
    "Y-90": Y_vals,
    "Radiation": radiation,
    "Shielded Radiation": radiation_shielded
})

filepath = "nuclear-decay-simulation/data/nuclear_decay_results.csv"

df.to_csv(filepath, index=False)

print("Saved!")
