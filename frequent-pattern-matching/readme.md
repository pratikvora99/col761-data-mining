# Frequent Pattern Matching<br>
Frequent pattern matching is the process of identifying patterns or sequences that appear often within a dataset, commonly used in data mining to discover associations, correlations, or structures hidden in large sets of data.<br>

Here is an implementation of two popular algorithms: Apriori and FP-Tree in C++. <br>

## Files<br>
File Name | Description
--- | ---
run.sh | Shell-script to run the code<br>
apriori_final.cpp  |   Implementation of Apriori algorithm<br>
common.lib        |      a bash library for storing constants<br>
compile.sh       |     Shell-script to compile the cpp code<br>
fp_tree_final.cpp   |    Final implementation(s) of FP-Tree algorithm<br>
plot.py         |       python code for plotting the runtime vs support-threshold graph<br>
report.pdf     |        explanation of the results obtained<br>
runtime-plot.png    |    the plot of runtimes for different support-thresholds<br>

## How to run the code<br>
To compile the code: run `sh compile.sh`. This should the necessary executable files according to the environment of the C++ installation. <br>

Now to find the frequent patterns using apriori/fp-tree algorithm: run `sh run.sh -algorithm -dataset -supportThresold -outFile`<br>
Here, algorithms are apriori or fptree.<br>
supportThreshold should be a float value ranging between 0 and 1.<br>

## Plot the run-time graph comparing the two algorithms.
In order to create the graph: run `sh run.sh -plot -dataset` <br>
This will plot a graph of runtimes of both the algorithms at various support thresholds: 5%, 10%, 25%, 50% and 75%.
