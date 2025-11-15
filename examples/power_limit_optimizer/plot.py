import matplotlib.pyplot as plt
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("input", type=str, help="Path to the power limit profile JSON file.")
args = parser.parse_args()

with open(args.input, "r") as f:
    profile_data = json.load(f)
output_prefix = args.input.replace(".json", "")
with open(args.input.replace(".json", ".meta.json"), "r") as f:
    metadata = json.load(f)

profile_steps = metadata["profile_steps"]
total_steps = metadata["steps_per_epoch"]
scaled_profile = []

fig, (ax1, ax2, ax4) = plt.subplots(3, 1, figsize=(8, 12))

# Plot GPU frequencies over time for each power limit
for measurement in profile_data["measurements"]:
    power_limit = measurement["power_limit"]
    frequency_data = measurement["frequency"]
    energy = measurement["energy"]
    time = measurement["time"]

    scaled_profile.append((power_limit, energy / profile_steps * total_steps, time / profile_steps * total_steps))

    for gpu_index, freq_list in frequency_data.items():
        times = [t for t, f in freq_list]
        freqs = [f for t, f in freq_list]
        ax1.plot(times, freqs, label=f"GPU {gpu_index}, Power Limit {power_limit} W")

ax1.set_title(f"GPU Frequencies over Time (Profiled Steps: {profile_steps})")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Frequency (MHz)")
ax1.legend()
ax1.grid(True)


# Plot energy and time vs. power limit (use twin y-axes)
scaled_profile.sort(key=lambda x: x[0])
powers = [p for p, e, t in scaled_profile]
energies = [e for p, e, t in scaled_profile]
times = [t for p, e, t in scaled_profile]
avg_powers = [e / t for p, e, t in scaled_profile]

ax2.plot(powers, energies, marker='o', color='tab:blue', label='Energy (J)')
ax2.set_xlabel('Power Limit (W)')
ax2.set_ylabel('Energy (J)', color='tab:blue')
ax2.set_title('Energy and Time per Epoch vs Power Limit (Scaled to Full Epoch)')

ax3 = ax2.twinx()
ax3.plot(powers, times, marker='o', color='tab:orange', label='Time (s)')
ax3.set_ylabel('Time (s)', color='tab:orange')

ax4.plot(powers, avg_powers, marker='o', color='tab:green', label='Average Power (W)')
ax4.set_xlabel('Power Limit (W)')
ax4.set_ylabel('Average Power (W)', color='tab:green')
ax4.set_title('Average Power vs Power Limit')

fig.tight_layout()
fig.savefig(f"{output_prefix}.png")
