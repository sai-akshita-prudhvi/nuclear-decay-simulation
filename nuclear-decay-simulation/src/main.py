import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    figures_dir = project_root / "figures"
    data_dir = project_root / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    dt = 1.0  # 1 step = 1 day
    time_steps = 2000
    time_days = [0.0]

    isotopes = {
        "Cs-137": {"half_life": 30.17 * 365.25, "daughter": "Ba-137m", "energy": 6.0},
        "Ba-137m": {"half_life": 2.6 / 24, "daughter": "Ba-137", "energy": 0.5},
        "Ba-137": {"half_life": None, "daughter": None, "energy": 0.0},
        "Sr-90": {"half_life": 28.8 * 365.25, "daughter": "Y-90", "energy": 5.0},
        "Y-90": {"half_life": 64 / 24, "daughter": "Zr-90", "energy": 7.0},
        "Zr-90": {"half_life": None, "daughter": None, "energy": 0.0},
    }

    N = {
        "Cs-137": 1000,
        "Ba-137m": 0,
        "Ba-137": 0,
        "Sr-90": 500,
        "Y-90": 0,
        "Zr-90": 0,
    }

    mu = 0.12
    shield_thickness = 10
    shielding_thicknesses = [0, 2, 5, 10, 15, 20]

    Cs_vals = [N["Cs-137"]]
    Sr_vals = [N["Sr-90"]]
    Y_vals = [N["Y-90"]]
    radiation = [0.0]
    radiation_shielded = [0.0]

    initial_radiation_source = N["Cs-137"] + N["Sr-90"] + N["Y-90"]
    final_radiation = [
        initial_radiation_source * np.exp(-mu * thickness)
        for thickness in shielding_thicknesses
    ]

    for t in range(1, time_steps + 1):
        current_time = t * dt
        time_days.append(current_time)

        new_N = N.copy()
        step_radiation = 0.0

        for iso, count in N.items():
            data = isotopes[iso]
            if data["half_life"] is None or count <= 0:
                continue

            lam = np.log(2) / data["half_life"]
            p_decay = 1 - np.exp(-lam * dt)
            decays = np.random.binomial(count, p_decay)

            daughter = data["daughter"]
            new_N[iso] -= decays
            if daughter is not None:
                new_N[daughter] += decays

            step_radiation += decays * data["energy"]

        N = new_N
        Cs_vals.append(N["Cs-137"])
        Sr_vals.append(N["Sr-90"])
        Y_vals.append(N["Y-90"])
        radiation.append(step_radiation)
        radiation_shielded.append(step_radiation * np.exp(-mu * shield_thickness))

    plt.figure(figsize=(9, 5))
    plt.plot(time_days, Cs_vals, label="Cs-137")
    plt.plot(time_days, Sr_vals, label="Sr-90")
    plt.plot(time_days, Y_vals, label="Y-90")
    plt.title("Isotope Populations Over Time (Linear Scale)")
    plt.xlabel("Time (days)")
    plt.ylabel("Atoms")
    plt.grid(True)
    plt.legend()
    plt.savefig(figures_dir / "isotope_decay_linear.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.semilogy(time_days, Cs_vals, label="Cs-137")
    plt.semilogy(time_days, Sr_vals, label="Sr-90")
    plt.semilogy(time_days, Y_vals, label="Y-90")
    plt.title("Isotope Populations Over Time (Log Scale)")
    plt.xlabel("Time (days)")
    plt.ylabel("Atoms (log scale)")
    plt.legend()
    plt.savefig(figures_dir / "isotope_decay_log.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.semilogy(time_days, radiation, label="Raw Radiation")
    plt.semilogy(time_days, radiation_shielded, label="Shielded Radiation")
    plt.title("Radiation Output (Log Scale)")
    plt.xlabel("Time (days)")
    plt.ylabel("Intensity (log scale)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(figures_dir / "radiation_output.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(shielding_thicknesses, final_radiation, marker="o", linewidth=2)
    plt.title("Effect of Shielding Thickness on Radiation")
    plt.xlabel("Shield Thickness (cm)")
    plt.ylabel("Detected Radiation")
    plt.grid(True)
    plt.savefig(figures_dir / "shielding_experiment.png", dpi=300, bbox_inches="tight")
    plt.show()

    df = pd.DataFrame(
        {
            "Time (days)": time_days,
            "Cs-137": Cs_vals,
            "Sr-90": Sr_vals,
            "Y-90": Y_vals,
            "Radiation": radiation,
            "Shielded Radiation": radiation_shielded,
        }
    )
    df.to_csv(data_dir / "nuclear_decay_results.csv", index=False)

    shield_df = pd.DataFrame(
        {
            "Shield Thickness (cm)": shielding_thicknesses,
            "Detected Radiation": final_radiation,
        }
    )
    shield_df.to_csv(data_dir / "shielding_experiment.csv", index=False)

    print("Saved simulation outputs to:")
    print(f" - {figures_dir}")
    print(f" - {data_dir}")


if __name__ == "__main__":
    main()
