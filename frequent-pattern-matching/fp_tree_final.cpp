#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
using namespace std;
#define ui unsigned int

double THRESHOlD = 0.3;         // Support Threshold
ui totalTransactions = 0;       // Total number of transactions in the database
map<ui,ui> itemFreqMap;
map<string,ui> itemToIDMap;     // To map item to its unique numeric id
map<ui,string> idToItemMap;     // To map the unique numeric id to its item
ui ID = 1;

/**
Class definition for an FP Tree Node:
    1. unsigned int itemID      : Item ID
    2. unsigned int freq        : Frequency of the item node
    3. vector<FPTree*> children : Vector of pointers pointing to children of the node
    4. FPTree* parent           : Pointer for the immediate parent of the node
    5. FPTree* next             : Pointer to the next node containing same item
*/
class FPTree{
public:
    ui itemID, freq;
    vector<FPTree*> children;
    FPTree *parent, *next;

    FPTree(ui itemID){
        // Initializing the node.
        this->itemID = itemID;
        freq = 1;
        next = parent = NULL;
    }

    void insert_children(FPTree* node){
        // Appends argument: 'node' to the children vector of the reference node.
        children.push_back(node);
    }

    FPTree* get_child(ui itemID){
        // Finds if the reference node has an immediate child with tag: item. If not, returns NULL.
        for(FPTree* child: children){
            if(child->itemID == itemID){
                return child;
            }
        }
        return NULL;
    }

    vector<FPTree*> get_parents(){
        // Forms a vector of ancestors of the reference node. The vector will begin with the immediate parent.
        FPTree* curParent = this->parent;
        vector<FPTree*> parentsVector;
        while(curParent!=NULL && curParent->itemID!=-1){
            parentsVector.push_back(curParent);
            curParent = curParent->parent;
        }
        return parentsVector;
    }

    void inc_freq(){
        // Increments frequency of the node.
        freq++;
    }
};

FPTree* createFPTreeNode(ui itemID){
    // Creates a new FP Tree Node. Returns reference to the newly created node.
    FPTree* node = new FPTree(itemID);
    return node;
}

void add_to_headTable(map<ui, FPTree*>& headTable, FPTree* node){
    // Links the argument:node to the chain in headTable
    node->next = headTable[node->itemID];
    headTable[node->itemID] = node;
}

vector<pair<FPTree*,ui>> get_nodes(map<ui,FPTree*> headTable, ui itemID){
    // Fetches the node chain linked to the headTable and corresponding to the argument: "item"
    vector<pair<FPTree*,ui>> nodes;
    FPTree* cur = headTable[itemID];
    while(cur){
        nodes.push_back(make_pair(cur,cur->freq));
        cur = cur->next;
    }
    return nodes;
}

vector<vector<string>> performConditionalMining(vector<pair<FPTree*,ui>> nodesFreqVector){
    
    // Performs Conditional Mining using the generated FP Tree
    vector<vector<string>> itemsets;        // Final vector of frequent itemsets to be returned
    ui itemID = nodesFreqVector[0].first->itemID;
    ui freq = 0;

    // Calculating the support of the nodes
    for(ui nodeIter=0; nodeIter<nodesFreqVector.size();nodeIter++){
        freq += nodesFreqVector[nodeIter].second;
    }
    if(freq*1.0/totalTransactions < THRESHOlD){
        // If the support of an itemset is less than threshold
        return itemsets;
    }
    
    itemsets.push_back({idToItemMap[itemID]});   // Pushing the current node item as an itemset, since its frequency is higher than threshold

    // Parent map to store parents of the current node mapped by their item, along with their frequency with the current node.
    map<ui, vector<pair<FPTree*,ui>>> parentMap;
    for(ui nodeIter=0; nodeIter<nodesFreqVector.size(); nodeIter++){
        FPTree* node = nodesFreqVector[nodeIter].first;
        ui nodeFreq = nodesFreqVector[nodeIter].second;
        vector<FPTree*> parents = node->get_parents();
        for(FPTree* parent: parents){
            // Iterating over parents and mapping them onto parentMap
            if(parentMap.count(parent->itemID) == 0){ 
                vector<pair<FPTree*,ui>> newParentFreqVector;
                newParentFreqVector.push_back(make_pair(parent,nodeFreq));
                parentMap[parent->itemID] = newParentFreqVector;
            }else{
                vector<pair<FPTree*,ui>> parentFreqVector = parentMap[parent->itemID];
                bool sameParentNodeFound = false;
                // Checking if there are multiple nodes with same parent node. If so, only one parent node will be maintained in map.
                for(int parentIter=0; parentIter<parentFreqVector.size(); parentIter++){
                    if(parentFreqVector[parentIter].first == parent){
                        parentFreqVector[parentIter].second += nodeFreq;
                        sameParentNodeFound = true;
                        break;
                    }
                }
                if(!sameParentNodeFound){
                    parentFreqVector.push_back(make_pair(parent,nodeFreq));
                }
                parentMap[parent->itemID] = parentFreqVector;
            }
        }
    }
    for(auto x: parentMap){
        // Calling performConditionalMining() recursively over parents of node in bottom-up approach
        vector<vector<string>> newItemsets = performConditionalMining(x.second);
        for(vector<string> itemset: newItemsets){
            /* 
            Adding self-item to the itemsets received from the parents. 
                Eg: If the current node has item 1 and receives {2}, {3} and {2,3} as itemsets, 
                    the resultant itemset will be:
                    {1,2} ,{1,3} and {1,2,3} and {1} - which was added earlier in the function.
            */
            itemset.push_back(idToItemMap[itemID]);
            sort(itemset.begin(), itemset.end());
            itemsets.push_back(itemset);
        }
    }
    return itemsets;
}
bool sortBySecondAsc(const pair<ui,ui> &p1, const pair<ui,ui> &p2){
    if(p1.second == p2.second)
        return p1.first < p2.first;
    return p1.second < p2.second;
}
bool sortByFreqMapDesc(ui item1, ui item2){
    if(itemFreqMap[item1] == itemFreqMap[item2])
        return item1 < item2;
    return itemFreqMap[item1] > itemFreqMap[item2];
}
int main(int argc, char** argv){

    // Fetching threshold and paths from command line arguments.
    THRESHOlD = argc>1 ? stod(argv[1])/100 : 0.3;
    string path_to_dataset = argc>2 ? argv[2] : "test1.dat";
    string path_to_output = argc>3 ? argv[3] : "out.dat";

    string itemsetStr;
    ifstream datasetFile(path_to_dataset);  // Opening dataset file in read-only mode

    while(getline(datasetFile, itemsetStr)){
        totalTransactions++;    // Calculating number of transactions
        stringstream itemsetStrStream(itemsetStr);
        string item;
        while(itemsetStrStream >> item){
            ui itemID = itemToIDMap[item];
            if(!itemID){
                itemToIDMap[item] = itemID = ID;
                idToItemMap[ID] = item;
                ID++;
            }
            itemFreqMap[itemID]++;
        }
    }
    vector<pair<ui,ui>> itemFreqVector;
    for(auto itemFreqEntry: itemFreqMap){
        if((itemFreqEntry.second*1.0/totalTransactions) >= THRESHOlD)
            itemFreqVector.push_back(make_pair(itemFreqEntry.first, itemFreqEntry.second));
    }
    sort(itemFreqVector.begin(), itemFreqVector.end(),sortBySecondAsc);

    // FP Tree Generation. Creating the first root node. 
    FPTree* root = createFPTreeNode(-1);
    map<ui,FPTree*> headTable;  // To store the link list of nodes with same item

    datasetFile.clear();
    datasetFile.seekg(0, datasetFile.beg);
    while(itemFreqVector.size() && getline(datasetFile, itemsetStr)){
        stringstream itemsetStrStream(itemsetStr);
        string item;
        vector<ui> itemsetVector;
        while(itemsetStrStream >> item){
            ui itemID = itemToIDMap[item];
            if((itemFreqMap[itemID]*1.0/totalTransactions) >= THRESHOlD)
                itemsetVector.push_back(itemID);
        }
        sort(itemsetVector.begin(), itemsetVector.end(), sortByFreqMapDesc);    // Sorting itemsetVector in descending lexicographic order
        
        // Pushing items from the itemsetVector into FP Tree
        FPTree* curNode = root;
        for(ui itemID: itemsetVector){
            FPTree* nextNode = curNode->get_child(itemID);
            if(!nextNode){
                nextNode = createFPTreeNode(itemID);
                curNode->insert_children(nextNode);
                nextNode->parent = curNode;
                add_to_headTable(headTable, nextNode);
            }else{
                nextNode->inc_freq();
            }
            curNode = nextNode;
        }
    }
    if(datasetFile.is_open())
        datasetFile.close();
    
    // Frequent Data set mining
    vector<vector<string>> freqItemsets;
    for(auto itemFreqPair: itemFreqVector){
        // For every item in itemVector, fetch all its nodes present in FP-Tree using headTable
        ui itemID = itemFreqPair.first;
        vector<pair<FPTree*,ui>> nodesFreqVector = get_nodes(headTable, itemID);

        // Calling performConditionalMining() on the vector of <nodes,frequency>
        vector<vector<string>> newFrequentItemsets = performConditionalMining(nodesFreqVector);
        freqItemsets.insert(freqItemsets.end(), newFrequentItemsets.begin(), newFrequentItemsets.end());
    }    
    
    ofstream outputFile(path_to_output); // Opening output file in write mode
    sort(freqItemsets.begin(), freqItemsets.end());
    for(vector<string> itemsets: freqItemsets){
        int first=1;
        for(string item: itemsets){
            if(!first)
                outputFile << " ";
            first = 0;
            outputFile << item;
        }
        outputFile << endl;
    }
    if(outputFile.is_open())
        outputFile.close();

    return 0;
}
