# Subgraph Isomorphism

Files:<br>
Frequent Subgraph Mining:<br> 
1. **run.sh**: To load modules for question 1 and run run.py
2. **run.py**: Calls the preprocessor.py and binaries on formatted preprocessed data
3. **preprocessor.py**: preprocesses the dataset into the form usable for each binaries
4. **gaston**: Gaston binary
5. **fsg**: FSG binary
6. **gSpan-64**: gSpan binary

Subgraph Isomorphism:<br>
1. **index.sh**: To execute index on the dataset provided using gSpan and then to create index using index.py
2. **gSpan-64**: Binary used to fetch all the subgraphs for a given support
3. **index.py**: Creates index on the provided dataset
4. **preprocessor.py**: Preprocesses the data to make it ready for gSpan
5. **postprocessor.py**: The frequent pattern file obtained as output is formatted to reinstate lost vertex labels and transaction IDs.
6. **query.sh**: Loads necessary modules, unzips boost library and runs query.cpp
7. **query.cpp**: Loads the index, asks for query file and runs index-eliminated subgraph isomorphism on dataset
8. **boost.zip**: Boost Library for Subgraph Isomorphism

K-Means clustering:<br>
1. **elbow_plot.sh**: Loads modules and runs run.py
2. **run.py**: For values of k from 1 to 15, runs KMeans clustering algorithm to plot elbow-graph.

Code Execution Instructions:<br>
Frequent Subgraph Mining:<br>
`sh run.sh \<dataset\>`<br>

Subgraph Isomorphism:<br>
`sh index.sh \<dataset\>`<br>
`sh query.sh`<br>

K-Means clustering: <br>
`sh elbow_plot.sh \<dataset\> \<dimension\> \<rollNo\>`<br>
`python run.py \<dataset\> \<dimension\> \<rollNo\>`<br>
