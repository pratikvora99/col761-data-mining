# col761
Everything including and upto the final submission for HW2<br>

Members:<br>
* **Pratik Vora: 2022AIB2687** (Contribution: 67%) (Q1 & Q2)
* **Dhruvil Sheth: 2022AIB2689** (Contribution: 33%) (Q3)

Files Bundled:<br>
Q1:<br> 
1. **q1.sh**: To load modules for question 1 and run q1_run.py
2. **q1_run.py**: Calls the preprocessor.py and binaries on formatted preprocessed data
3. **preprocessor.py**: preprocesses the dataset into the form usable for each binaries
4. **gaston**: Gaston binary
5. **fsg**: FSG binary
6. **gSpan-64**: gSpan binary

Q2:<br>
1. **index.sh**: To execute index on the dataset provided using gSpan and then to create index using index.py
2. **gSpan-64**: Binary used to fetch all the subgraphs for a given support
3. **index.py**: Creates index on the provided dataset
4. **preprocessor.py**: Preprocesses the data to make it ready for gSpan
5. **postprocessor.py**: The frequent pattern file obtained as output is formatted to reinstate lost vertex labels and transaction IDs.
6. **query.sh**: Loads necessary modules, unzips boost library and runs query.cpp
7. **query.cpp**: Loads the index, asks for query file and runs index-eliminated subgraph isomorphism on dataset
8. **boost.zip**: Boost Library for Subgraph Isomorphism

Q3:<br>
1. **elbow_plot.sh**: Loads modules and runs q1.py
2. **q1.py**: For values of k from 1 to 15, runs KMeans clustering algorithm to plot elbow-graph.

Code Execution Instructions:<br>
Q1:<br>
sh q1.sh \<dataset\><br>

Q2:<br>
sh index.sh \<dataset\><br>
sh query.sh<br>

Q3: <br>
sh elbow_plot.sh \<dataset\> \<dimension\> \<rollNo\><br>
python q3.py \<dataset\> \<dimension\> \<rollNo\><br>
