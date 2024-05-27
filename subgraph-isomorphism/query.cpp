#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/vf2_sub_graph_iso.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <queue>

using namespace std;
using namespace boost;

vector<string> split_string(string str){
    stringstream ss(str);
    vector<string> splits;
    string split;
    while(ss >> split)
        splits.push_back(split);
    return splits;
}

template <typename Graph1, typename Graph2>
class my_call_back {
public:
    my_call_back(const Graph1& graph1, const Graph2& graph2) : graph1_(graph1), graph2_(graph2) {}

    template <typename CorrespondenceMap1To2, typename CorrespondenceMap2To1>
    bool operator()(CorrespondenceMap1To2 f, CorrespondenceMap2To1) {
        return true;
    }
private:
    const Graph1& graph1_;
    const Graph2& graph2_;
};

int main()
{
    int N = 50;
    typedef property<edge_name_t, string> edge_property;
    typedef property<vertex_name_t, string, property<vertex_index_t, int> > vertex_property;
    typedef adjacency_list<vecS, vecS, bidirectionalS, vertex_property, edge_property> graph_type;

    cout << "--- Loading Index ---" << endl;
    string dataset_path,subgraphCountStr;
    ifstream statFile("index_stats.txt");
    getline(statFile,dataset_path);
    getline(statFile,subgraphCountStr);
    // cout << dataset_path << endl;
    cout << "Number of subgraphs found: " << subgraphCountStr << endl;
    int subgraphCount = stoi(subgraphCountStr);
    if(statFile.is_open())
        statFile.close();
    
    map<string, graph_type> transactionGraphs;
    vector<string> transactionIds;
    ifstream transactionFile(dataset_path);
    string line;
    int vcount = -1, ecount = -1;
    int transactionIter = 0;
    while(getline(transactionFile, line)){
        trim(line);
        if(line == "")
            continue;
        if(line[0] == '#'){
            graph_type transactionGraph;
            string tid = line.substr(1);
            getline(transactionFile,line); trim(line);
            // cout << "transaction file v line: (" << line << ")" << endl;
            vcount = stoi(line);
            while(vcount--){
                getline(transactionFile,line); trim(line);
                add_vertex(vertex_property(line), transactionGraph);
            }
            getline(transactionFile,line); trim(line);
            // cout << "transaction file e line: (" << line << ")" << endl;
            ecount = stoi(line);
            while(ecount--){
                getline(transactionFile,line); trim(line);
                vector<string> edgeSplits = split_string(line);
                // cout << "transaction edgeSplits: (" << edgeSplits[0] << "," << edgeSplits[1] << ")" << endl;
                add_edge(stoi(edgeSplits[0]), stoi(edgeSplits[1]), edge_property(edgeSplits[2]), transactionGraph);
            }
            transactionGraphs[tid] = transactionGraph;
            transactionIds.push_back(tid);
            transactionGraph.clear();
        }
    }
    if(transactionFile.is_open())
        transactionFile.close();

    cout << "Number of transactions scanned: " << transactionIds.size() << endl;

    vector<graph_type> tempSubgraphs;
    priority_queue<pair<int, int>> subgraphHeap;

    ifstream indexFile("index_raw.fp");  // Opening mined frequent subset file
    int subgraphIter = 0;
    map<string,vector<int>> transactionIndices;
    while(getline(indexFile, line)){
        trim(line);
        if(line == ""){
            continue;
        }
        if(line[0] == 't'){
            graph_type subgraph;
            vector<string> sid_list = split_string(line);
            int support = stoi(sid_list[4]);
            getline(indexFile, line); trim(line);
            while(line != ""){
                if(line[0] == 'v'){
                    vector<string> vertexSplits = split_string(line);
                    add_vertex(vertex_property(vertexSplits[2]), subgraph);
                } else if(line[0] == 'e'){
                    vector<string> edgeSplits = split_string(line);
                    // cout << "subgraph edgeSplits: (" << edgeSplits[1] << "," << edgeSplits[2] << ")" << endl;
                    add_edge(stoi(edgeSplits[1]), stoi(edgeSplits[2]), edge_property(edgeSplits[3]), subgraph);
                } 
                else if(line[0] == 'x'){
                    vector<string> tid_list = split_string(line);
                    for(int i=1;i<tid_list.size();i++){
                        string tIter = tid_list[i];
                        if(transactionIndices.find(tIter) == transactionIndices.end())
                            transactionIndices[tIter] = vector<int>(subgraphCount);
                        transactionIndices[tIter][subgraphIter] = 1;
                    }
                }
                getline(indexFile, line);
            }
            subgraphHeap.push(make_pair(support,subgraphIter));
            tempSubgraphs.push_back(subgraph);
            subgraph.clear();
            subgraphIter++;
            if(subgraphIter >= subgraphCount) break;
        }
    }
    if(indexFile.is_open())
        indexFile.close();
    
    subgraphCount = min(N, subgraphCount);
    vector<int> subgraphIters;
    for(int i=0;i<subgraphCount;i++){
        subgraphIters.push_back(subgraphHeap.top().second);
        subgraphHeap.pop();
    }

    vector<graph_type> subgraphs;
    for(int i=0;i<subgraphCount;i++){
        subgraphs.push_back(tempSubgraphs[subgraphIters[i]]);
    }

    for(auto item:transactionIndices){
        string tid = item.first;
        vector<int> oldIndex = item.second;
        vector<int> newIndex(subgraphCount);
        for(int i=0;i<subgraphCount;i++)
            newIndex[i] = oldIndex[subgraphIters[i]];
        transactionIndices[tid] = newIndex;
    }

    auto startIndex = chrono::high_resolution_clock::now();
    for(auto item:transactionIndices){
        string tid = item.first;
        vector<int> tIndex = item.second;
        // if(tid != "42614805") continue;
        for(int i=0;i<subgraphCount;i++){
            if(tIndex[i] == 0){
                graph_type subgraph = subgraphs[i];
                graph_type transactionGraph = transactionGraphs[tid];
                my_call_back<graph_type, graph_type> callback(subgraph, transactionGraph);
                if(vf2_subgraph_iso(subgraph, transactionGraph, callback)){
                    // cout << "yes" << endl;
                    tIndex[i] = 1;
                }
            }
        }
        transactionIndices[tid] = tIndex;
    }
    auto endIndex = chrono::high_resolution_clock::now();
    auto durationIndex = chrono::duration_cast<chrono::milliseconds>(endIndex- startIndex);
    cout << "Time taken to execute the indexing: " << durationIndex.count() << " milliseconds" << endl << endl;
    // for(auto x:transactionIndices){
    //     string id = x.first;
    //     cout << "Transaction: " << id << endl;
    //     for(int i=0;i<subgraphCount;i++)
    //         cout << x.second[i] << " ";
    //     cout << endl;
    // }
    // return 0;
    cout << "Please provide path to the query file: ";
    string queryPath;
    cin >> queryPath;

    ifstream queryFile(queryPath);
    graph_type queryGraph;
    vector<vector<string>> finalLists;
    auto total_duration = 0;
    int qcount = 1;
    while(getline(queryFile, line)){
        trim(line);
        if(line == "")
            continue;
        if(line[0] == '#'){
            cout << "--Processing query " << line.substr(1) << " --" << endl;
            auto start = chrono::high_resolution_clock::now();
            getline(queryFile,line); trim(line);
            // cout << "query v line: " << line << endl;
            vcount = stoi(line);
            while(vcount--){
                getline(queryFile,line);
                trim(line);
                add_vertex(vertex_property(line), queryGraph);
            }
            getline(queryFile,line); trim(line);
            // cout << "query e line: " << line << endl;
            ecount = stoi(line);
            while(ecount--){
                getline(queryFile,line); trim(line);
                vector<string> edgeSplits = split_string(line);
                // cout << "query edgeSplits: (" << edgeSplits[0] << "," << edgeSplits[1] << "," << edgeSplits[2] << ")" << endl;
                add_edge(stoi(edgeSplits[0]), stoi(edgeSplits[1]), edge_property(edgeSplits[2]), queryGraph);
            }

            vector<int> queryIndex(subgraphCount);
            for(int i=0;i<subgraphCount;i++){
                graph_type subgraph = subgraphs[i];

                my_call_back<graph_type, graph_type> callback(subgraph, queryGraph);
                if(subgraph.vertex_set().size()<=queryGraph.vertex_set().size() && 
                        subgraph.m_edges.size()<=queryGraph.m_edges.size() &&
                            vf2_subgraph_iso(subgraph, queryGraph, callback)){
                    // cout << "subgraph " << i << endl;
                    queryIndex[i] = 1;
                }
                // cout << queryIndex[i] << " ";
            }
            // cout << endl;
            set<string> removableTransactions;
            for(auto item: transactionIndices){
                string tIter = item.first;
                vector<int> tIndex = item.second;
                // if(tIter == "42614805"){
                //     for(int i=0;i<subgraphCount;i++){
                //         cout << tIndex[i] << " ";
                //     }
                //     cout << endl;
                // }
                for(int i=0;i<subgraphCount;i++){
                    if(queryIndex[i]==1 && tIndex[i]==0){
                        removableTransactions.insert(tIter);
                        break;
                    }
                }
            }
            // cout << "Number of transactions filtered using indexes: " << removableTransactions.size() << endl;
            vector<string> finalList;
            for(auto item: transactionGraphs){
                string tid = item.first;
                if(removableTransactions.find(tid) != removableTransactions.end())
                    continue;
                graph_type transactionGraph = item.second;

                // cout << "Checking for: " << transactionIds[i] << endl;
                my_call_back<graph_type, graph_type> callback(queryGraph, transactionGraph);
                if(queryGraph.vertex_set().size()<=transactionGraph.vertex_set().size() && 
                        queryGraph.m_edges.size()<=transactionGraph.m_edges.size() && 
                            vf2_subgraph_iso(queryGraph, transactionGraph, callback))
                    finalList.push_back(tid);
            }
            sort(finalList.begin(), finalList.end());
            // for(string tid: finalList)
            //     cout << tid << " ";
            // cout << endl;
            finalLists.push_back(finalList);
            queryGraph.clear();
            qcount++;
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
            total_duration += duration.count();
            cout << "Time taken to execute the query: " << duration.count() << " milliseconds" << endl << endl;
        }
    }
    cout << "Total duration to process all the queries: " << total_duration << " milliseconds" << endl;
    if(queryFile.is_open())
        queryFile.close();
    
    ofstream outputFile("output_AIB222687.txt"); // Opening output file in write mode
    for(vector<string> finalList: finalLists){
        int first=1;
        for(string tid: finalList){
            if(!first)
                outputFile << "\t";
            first = 0;
            outputFile << tid;
        }
        outputFile << endl;
    }
    if(outputFile.is_open())
        outputFile.close();
    cout << "Final result saved in output_AIB222687.txt" << endl;
    return 0;
}