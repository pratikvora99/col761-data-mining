#include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std;
using namespace std::chrono;
map<string,long long> freqHashMap;

long long supportPercent = 98;
string fileName = "D:\\IITD\\Data Mining\\Assignment\\HW1\\test5.dat";
string opFile = "D:\\IITD\\Data Mining\\Assignment\\HW1\\outtest5.dat";

long long totalTransLines = 0;

ifstream transFile;
set<string> finalResult;


void createHashMap(string transLine)
{
	stringstream ss(transLine);
	string item;
	
	while(ss >> item)
	{
		freqHashMap[item]++;
	}
	
}


vector<vector<string>> pruneHashMap(long long support)
{ 
	vector<string> temp;
	vector<vector<string>> freqItemSet;

	for (auto it = freqHashMap.begin(); it!=freqHashMap.end();) 
	   {
	   		if(it->second<support)
	   		{
	   			freqHashMap.erase(it++);
	   		}
	   		else
	   		{
	   			temp.clear();
	   			temp.push_back(it->first);
	   			freqItemSet.push_back(temp);
	   			//globalFreqItemSet.push_back(temp);
	   			finalResult.insert(it->first);
	   			it++;
	   		}
	   	}

	return freqItemSet;
}


// void printFreqItemSet(vector<vector<string>> freqSet)
// {
// 	cout<<"Printing item set"<<endl;
// 	for(auto &vector: freqSet)
// 	{
// 		for(auto &string: vector)
// 		{
// 			cout<<string<<" ";
// 		}
// 		cout<<endl;
// 	}
// }

// void printFreqHashMap(map<string,long long> freqMap)
// {
// 	cout<<"Ptrinting Hash map"<<endl;
// 	for(auto &pair:freqMap)
// 	{
// 		cout<<"{"<<pair.first<<":"<<pair.second<<"}"<<endl;
// 	}
// }


vector<vector<string>> pruneCandidateSet(vector<vector<string>> candidateSet, vector<vector<string>> freqSet)
{
	vector<vector<string>> prunedCandidateSet;
	vector<string> temp;
	int m = candidateSet.size();
	int n,k=0;
	bool isPresent = false, shouldPrune = false;

	for(int i=0;i<m;i++)
	{
		shouldPrune = false;
		n = candidateSet[i].size();
		if(n<3)
			return candidateSet;
		for(int j=0;j<n;j++)
		{
			isPresent = false;

			temp = candidateSet[i];
			temp.erase(temp.begin()+k);
			k++;
			for(auto &vector:freqSet)
			{
				if(temp == vector)
				{
					isPresent = true;
					break;
				}
			}

			if(!isPresent)
			{
				shouldPrune = true;
				break;
			}


		}

		if(!shouldPrune)
			prunedCandidateSet.push_back(candidateSet[i]);
	}

	return prunedCandidateSet;
}


vector<vector<string>> mergeItemSet(vector<vector<string>> freqSet)
{
	vector<vector<string>> candidateSet;
	vector<string> temp1, temp2;

	int m = freqSet.size();

	for(int i=0;i<m-1;i++)
	{
		for(int j=i+1;j<m;j++)
		{
			temp1.clear();
			temp2.clear();

			temp1.insert(temp1.begin(), freqSet[i].begin(), --freqSet[i].end());
			temp2.insert(temp2.begin(), freqSet[j].begin(), --freqSet[j].end());
			if(temp1 == temp2)
			{
				temp1.clear();
				temp1.insert(temp1.begin(), freqSet[i].begin(), freqSet[i].end());
				temp1.push_back(*(--freqSet[j].end()));
				
				candidateSet.push_back(temp1);
			}
		}
	}


	candidateSet = pruneCandidateSet(candidateSet,freqSet);

	return candidateSet;
}


vector<string> splitStringToVector(string lineToSplit)
{
	stringstream ss(lineToSplit);
	istream_iterator<string> begin(ss);
	istream_iterator<string> end;

	vector<string> vect(begin, end);

	return vect;

}

vector<vector<string>> pruneFrequentSet(vector<vector<string>> candidateSet, long long minSup)
{
	vector<vector<string>> freqSet;
	int m = candidateSet.size();
	bool isPresent = true;
	long long ctr=0;
	string temp="";
	string transactionLine;
	vector<string> transaction;
	vector<long long> freqCount(m,0);
	long long ind = 0;


	transFile.clear();
	transFile.seekg(0, ios::beg);

	while ( getline (transFile, transactionLine) )
		{
	

		ind =0;
		transaction = splitStringToVector(transactionLine);
		for(auto &vector: candidateSet)
		{
				isPresent = true;

				
				for(auto &string: vector)
				{
					if(find(transaction.begin(), transaction.end(), string) == transaction.end())
					{
							isPresent = false;
							break;
					}
				}
				if(isPresent)
				{
					freqCount[ind]++;
				}
				ind++;

			}
		}

		for(int i=0;i<m;i++)
		{
			if(freqCount[i]>=minSup)
			{
				freqSet.push_back(candidateSet[i]);
				temp = "";
				for(int j=0; j<candidateSet[i].size();j++)
				{
					temp+=candidateSet[i][j]+" ";
				}
				finalResult.insert(temp);
			}
		}

	return freqSet;
}

void writeFinalResult(set<string> a)
{
	ofstream fileWrite(opFile);
	//cout<<"Writing in file"<<endl;
	for(auto &string:a)
	{
		fileWrite<<string<<"\n";
		//cout<<string<<endl;
	}
	fileWrite.close();
}

int main(int argc, char** argv)
{

	string transactionLine;
	long long support;

	supportPercent = argc>1 ? stod(argv[1]) : 30;
   	fileName = argc>2 ? argv[2] : "test1.dat";
	opFile = argc>3 ? argv[3] : "out.dat";


	vector<vector<string>> freqItemSet, candidateItemSet;

	 transFile.open(fileName);

	 while ( getline (transFile, transactionLine) )
	 {
	 	createHashMap(transactionLine);
	 	totalTransLines++;
		
	 }
	 support = ceil((supportPercent*totalTransLines*1.0)/100);
	 freqItemSet = pruneHashMap(support);

	 while(!freqItemSet.empty())
	 {
	 	candidateItemSet = mergeItemSet(freqItemSet);	 	
	 	freqItemSet.clear();
		freqItemSet = pruneFrequentSet(candidateItemSet, support);
	 }
	 transFile.close();
	 writeFinalResult(finalResult);

}