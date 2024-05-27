import time
import math
import os
import matplotlib.pyplot as plt
import sys

if __name__=="__main__":
    dataset_name = sys.argv[1]
    base_name = dataset_name.split(".")[0] + "_"
    out_file = "q1_AIB222687.png"

    supports = [95.0, 50.0, 25.0, 10.0, 5.0]
    runtimes = {"gaston":[], "gspan":[], "fsg":[]}
    transaction_count = 0

    with open(base_name+"gaston", "r") as fp:
        line = fp.readline()
        while(line != ""):
            if(line.__contains__("t #")):
                transaction_count+=1
            line = fp.readline()

    for support in supports:
        for algo in ["gaston", "gspan", "fsg"]:
            start = time.time()
            command = ""
            if algo == "gaston":
                command = "./gaston " + str(math.ceil(support*transaction_count/100.0)) + " " + base_name + algo
            elif algo == "gspan":
                command = "./gSpan-64 -f " + base_name+algo + " -s " + str(support/100.0) + " -o -m 1"
            else:
                command = "./fsg -s " + str(support) + " " + base_name+algo
                print(command)
            os.system(command)
            end = time.time()
            runtimes[algo].append(end-start)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
    ax.plot(supports, runtimes["gaston"], label="Gaston", color="green", marker='x')
    ax.plot(supports, runtimes["gspan"], label="gSpan", color="red", marker='x')
    ax.plot(supports, runtimes["fsg"], label="FSG", color="blue", marker='x')

    ax.set_title("Runtime vs Support-threshold")
    ax.set_ylabel("Runtime (in seconds)")
    ax.set_xlabel("Support-threshold (in percentage)")
    ax.set_xticks(supports)
    ax.legend()
    plt.savefig(out_file)
