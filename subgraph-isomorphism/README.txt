Members:
Pratik Vora: 2022AIB2687 (Contribution: 67%) (Q1 & Q2)
Dhruvil Sheth: 2022AIB2689 (Contribution: 33%) (Q3)

Files Bundled:

Q1:
q1.sh: To load modules for question 1 and run q1_run.py
q1_run.py: Calls the preprocessor.py and binaries on formatted preprocessed data (Saves the plot as q1_AIB222687.png)
preprocessor.py: preprocesses the dataset into the form usable for each binaries
gaston: Gaston binary
fsg: FSG binary
gSpan-64: gSpan binary

Q2:
index.sh: To execute index on the dataset provided using gSpan and then to create index using index.py
gSpan-64: Binary used to fetch all the subgraphs for a given support
index.py: Creates index on the provided dataset
preprocessor.py: Preprocesses the data to make it ready for gSpan
postprocessor.py: The frequent pattern file obtained as output is formatted to reinstate lost vertex labels and transaction IDs.
query.sh: Loads necessary modules, unzips boost library and runs query.cpp
query.cpp: Loads the index, asks for query file and runs index-eliminated subgraph isomorphism on dataset
boost.zip: Boost Library for Subgraph Isomorphism

Q3:
elbow_plot.sh: Loads modules and runs q1.py
q1.py: For values of k from 1 to 15, runs KMeans clustering algorithm to plot elbow-graph.

Code Execution Instructions:

Q1:
sh q1.sh <dataset>
Output: q1_AIB222687.png

Q2:
sh index.sh <dataset>
sh query.sh
Output: output_AIB222687.txt

Q3: 
sh elbow_plot.sh <dataset> <dimension> <rollNo>
python q3.py <dataset> <dimension> <rollNo>
