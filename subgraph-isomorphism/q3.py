from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import sys

def preprocessingData(df):
  scaler = MinMaxScaler()
  model=scaler.fit(df)
  scaled_data=model.transform(df)
  return scaled_data

def computeEuclideanError(df,clusterRange):
  err = list()
  for k in clusterRange:
    kMeans = KMeans(n_clusters = k)
    kMeans.fit(df)
    err.append(kMeans.inertia_)
  return err

def plot(clusterRange, euclideanError, plotName, dimension):
  plt.plot(clusterRange, euclideanError,c='r',marker='x')
  plt.xlabel("No of Clusters")
  plt.ylabel("Intra-cluster Euclidean error")
  plt.title('Elbow plot for k clusters and {} dimensions with Min Max Scaler'.format(dimension))
  plt.savefig(plotName)
  plt.xticks(clusterRange)
  plt.show()


def main():
  args = sys.argv
  if(len(args)>=4):
    dataset = args[1]
    dimension = args[2]
    rollNum = args[3]
    plotName = "q3_{}_{}.png".format(dimension,rollNum)
    
    df = pd.read_csv(dataset, header=None, sep = '\s+')
    df = preprocessingData(df)
    maxRangeK = 15
    clusterRange = range(1,maxRangeK+1)
    err = computeEuclideanError(df, clusterRange)
    #print(err)
    plot(clusterRange, err, plotName,dimension)

  else:
    raise Exception("Incorrect Number of Arguments")
  
if __name__ == "__main__":
  main()
