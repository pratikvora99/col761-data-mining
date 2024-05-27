from matplotlib import pyplot as plt
import numpy as np
import sys

def plot_data(data: np.ndarray, out_file: str):
    print(type(data))
    print(data)
    data = np.where(data < 0, np.nan, data)

    x = data[:,0]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
    ax.plot(x, data[:,1], label="Apriori", color="green")
    ax.plot(x, data[:,2], label="FP-Tree", color="red")

    ax.set_title("Runtime vs Support-threshold")
    ax.set_ylabel("Runtime (in seconds)")
    ax.set_xlabel("Support-threshold (in percentage)")
    ax.set_xticks(x)
    
    ax.legend()

    plt.savefig(out_file)
    print(f"Successfully generated file: {out_file}")

if __name__ == "__main__":
    data_file = sys.argv[1]
    out_file = sys.argv[2]
    if ".png" not in out_file:
        out_file = f"{out_file}.png"

    data = list()
    with open(data_file, 'r') as f:
        print(f"Plotting data from file: {data_file}")
        line = f.readline()
        while line != "":
            data.append([np.int32(i) for i in line.strip("\n").split()])
            line = f.readline()

    plot_data(data=np.array(data), out_file=out_file)
