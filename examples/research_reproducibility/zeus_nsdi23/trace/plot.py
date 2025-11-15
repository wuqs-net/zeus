from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("--gpu", "-g", type=str, default="a40", help="GPU model name.")
parser.add_argument("--network", "-n", type=str, required=True, help="Network name.")
parser.add_argument("--batch_size", "-b", type=int, required=True, help="Batch size.")
parser.add_argument("--optimizer", "-o", type=str, default="adadelta", help="Optimizer name.")
args = parser.parse_args()

trace = f"summary_power_{args.gpu}.csv"
df = pd.read_csv(trace)
df = df[(df["network"] == args.network) & (df["batch_size"] == args.batch_size) & (df["optimizer"] == args.optimizer)]
df["energy_per_epoch"] = df["time_per_epoch"] * df["average_power"]


fig, (ax2, ax4) = plt.subplots(2, 1, figsize=(8, 8))

powers = df["power_limit"]
energies = df["energy_per_epoch"]
times = df["time_per_epoch"]

ax2.plot(powers, energies, marker='o', color='tab:blue', label='Energy (J)')
ax2.set_xlabel('Power Limit (W)')
ax2.set_ylabel('Energy (J)', color='tab:blue')
ax2.set_title('Energy and Time per Epoch vs Power Limit (from Zeus Trace)')

ax3 = ax2.twinx()
ax3.plot(powers, times, marker='o', color='tab:orange', label='Time (s)')
ax3.set_ylabel('Time (s)', color='tab:orange')

ax4.plot(powers, df["average_power"], marker='o', color='tab:green', label='Average Power (W)')
ax4.set_xlabel('Power Limit (W)')
ax4.set_ylabel('Average Power (W)', color='tab:green')
ax4.set_title('Average Power vs Power Limit (from Zeus Trace)')

fig.tight_layout()
fig.savefig(f"{args.network}_{args.optimizer}_{args.batch_size}.png")
