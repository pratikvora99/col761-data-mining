#!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
#!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
#!pip install -q torch-cluster -f https://pytorch-geometric.com/whl/torch-1.12.0+cu113.html

#!pip install -q torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.12.0+cu113.html
#!pip install -q torch-geometric
#!pip install -q torch-geometric-temporal

import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric_temporal.nn.recurrent import A3TGCN, A3TGCN2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

def get_adjacency(split_path, adj_path, label="train"):
  df_adj=pd.read_csv(adj_path)
  #print(list(df_adj.index))
  df_adj.to_csv("adj_task2_saved.csv", index=False)

  #print(pd.read_csv("adj_task1_saved.csv"))

  df_adj=pd.read_csv(adj_path)

  df2 = df_adj.set_index(df_adj.columns[0])

  splits = np.load(split_path)

  if label=="train":
    node_ids = splits["train_node_ids"]
  elif label=="val":
    node_ids = splits["val_node_ids"]
  elif label=="test":
    node_ids = splits["test_node_ids"]
  elif label=="whole":
    return df2.to_numpy()

  #train_node_ids = splits["train_node_ids"]
  #val_node_ids = splits["val_node_ids"]
  #test_node_ids = splits["test_node_ids"]

  return df2.loc[list(node_ids)][node_ids.astype(str)].to_numpy()

def get_edge_index_and_weights(adj_matrix):
  #print(adj_matrix.shape)
  adj_matrix_tensor = torch.tensor(adj_matrix, dtype = torch.long)
  edge_index, edge_weight = dense_to_sparse(adj_matrix_tensor)
  return edge_index, edge_weight.float()

def divide_data_into_windows(split_path, data_x_path, p, f, label="train"):

  df_data=pd.read_csv(data_x_path,  index_col=0)
  #print(np.array(list(df_data), dtype='int64'))

  splits = np.load(split_path)

  np.savez("graph_splits_task2_saved.npz", train_node_ids=splits["train_node_ids"], val_node_ids = splits["val_node_ids"], test_node_ids = splits["test_node_ids"])

  if label=="train":
    node_ids = splits["train_node_ids"]
  elif label=="val":
    node_ids = splits["val_node_ids"]
  elif label=="test":
    node_ids = splits["test_node_ids"]
  elif label == "whole":
    node_ids = np.array(list(df_data), dtype='int64')

  
  df_data = df_data[node_ids.astype(str)]

  temporal_dfs = []

  X = []
  Y = []

  for i in range(len(df_data)-p-f+1):
    temporal_dfs.append(df_data.iloc[i:i+p+f, :].to_numpy())
  
  #print(np.array(temporal_dfs).shape)
  for arr in temporal_dfs:
    X.append(arr[:p].T.reshape(-1, 1, arr[:p].T.shape[1]))
    Y.append(arr[p:p+f].T)

  return np.array(X), np.array(Y)

def get_mask(split_path, node_id_list, label="train"):

  splits = np.load(split_path)

  if label=="train":
    node_ids = splits["train_node_ids"]
  elif label=="val":
    node_ids = splits["val_node_ids"]
  elif label=="test":
    node_ids = splits["test_node_ids"]

  mask = np.array([True if i in node_ids else False for i in node_id_list])
  return mask


def get_device():
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    else:
        return torch.device("cpu")


############ prepare graph dataset ##############
def create_data(X, Y, edge_index, edge_weight, train_mask, val_mask, test_mask):
  data_list = []

  for i in range(X.shape[0]):
          node_features = torch.LongTensor(X[i].reshape(-1,1))

          x = node_features
          y = torch.LongTensor(Y[i].reshape(-1,1))

          data = Data (x=x, edge_index = edge_index, edge_attr = edge_weight.unsqueeze(1), train_mask=torch.BoolTensor(train_mask), val_mask=torch.BoolTensor(val_mask), test_mask=torch.BoolTensor(test_mask), y=y)
          data_list.append(data)

  return data_list

def visualize_graph(data):
  g = torch_geometric.utils.to_networkx(data, to_undirected=False)
  print(g.number_of_edges())
  print(g.number_of_nodes())
  nx.draw(g)

########################## Model ###############################
class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, f):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features, out_channels=16, batch_size = 128, periods=periods)
        self.linear = torch.nn.Linear(16, f)

    def forward(self, x, edge_index, edge_weight):
        h = self.tgnn(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


def training(model, train_dataset, n_epoch, optimizer, batch_size):
    device = get_device()
    
    model.to(device)

    input = np.array(train_dataset.features) 
    target = np.array(train_dataset.targets) 
    train_x = torch.from_numpy(input).type(torch.FloatTensor).to(device)
    target_tensor = torch.from_numpy(target).type(torch.FloatTensor).to(device) 
    train_dataset_new = torch.utils.data.TensorDataset(train_x, target_tensor)

    #train_dataset_new.to(device)
    
    train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=128, shuffle=True, drop_last=True)
    
    #train_loader.to(device)

    for snapshot in train_dataset:
      snapshot.to(device)
      static_edge_index = snapshot.edge_index
      static_edge_weight = snapshot.edge_attr
      break

    model.train()

    for epoch in tqdm(range(n_epoch)):
      step = 0
      loss_list = []
      for encoder_inputs, labels in train_loader:
          y_hat = model(encoder_inputs, static_edge_index.to(device), static_edge_weight.to(device))  
          loss = torch.mean(torch.abs(y_hat-labels))       
          #loss = loss_fn(y_hat, labels) 
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
          step= step+ 1
          loss_list.append(loss.item())
          #if step % 100 == 0 :
            #print(sum(loss_list)/len(loss_list))

      if(epoch%20 == 0 or epoch==n_epoch-1):
        print("\nEpoch {} train MAE: {:.4f}".format(epoch, sum(loss_list)/len(loss_list)))  
      
    return model

def plot_train_val_loss(train_loss, val_loss, n_epoch):
    epochs = np.arange(1, n_epoch+1, 1)
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)

    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.legend()
    plt.title("MAE v/s Epochs")
    plt.xlabel("Epoch#")
    plt.ylabel("MAE")
    plt.show()

def predict_during_test(model, test_data, f):
  device = get_device()
  #model = GNN(hidden_channels_1=16)
  #model = torch.load("task1_d1_graphconv_2_layers.model")
  model.to(device)
  #model.eval()

  for snapshot in test_data:
    #print(snapshot)
    static_edge_index = snapshot.edge_index
    static_edge_weight = snapshot.edge_attr
    break
  
  #print(static_edge_index)
  #print(static_edge_weight)

  input = np.array(test_data.features)
  target = np.array(test_data.targets)
  test_x_tensor = torch.from_numpy(input).type(torch.FloatTensor).to(device)
  test_target_tensor = torch.from_numpy(target).type(torch.FloatTensor).to(device)
  test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)

  test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size= 1, shuffle=False, drop_last=False)

  model.eval()
  output = []
  

  for x, y_true in test_loader:
    out = model(x, static_edge_index.to(device), static_edge_weight.to(device))

    output.append(out.cpu().detach().numpy()[0].T[:f, :])

  return np.array(output)

############################# MAIN ###############################################
task = sys.argv[1]
#task = "train"

if(task == "train"):
    p = int(sys.argv[2])
    f = int(sys.argv[3])
    train_data_path = sys.argv[4]
    adj_path = sys.argv[5]
    graph_splits_path = sys.argv[6]

    #p = 12
    #f = 12
    #train_data_path = "d2_X.csv"
    #adj_path = "d2_adj_mx.csv"
    #graph_splits_path = "d2_graph_splits.npz"

    ############# load data ####################

    """
    adj_mat, node_id_list = get_adjacency(adj_path)
    X, Y = get_train_val_test_nodes(graph_splits_path, train_data_path, node_id_list)

    adj_mat, node_id_list = get_adjacency(adj_path)
    edge_index, edge_weight = get_edge_index_and_weights(adj_mat)

    train_mask = get_mask(graph_splits_path, node_id_list, label="train")
    val_mask = get_mask(graph_splits_path, node_id_list, label="val")
    test_mask = get_mask(graph_splits_path, node_id_list, label="test")

    data_list = create_data(X, Y, edge_index, edge_weight, train_mask, val_mask, test_mask)
    #print(len(data_list))
    """
    X_train, Y_train = divide_data_into_windows(graph_splits_path, train_data_path, p, f, label="train")

    train_adj_mat = get_adjacency(graph_splits_path, adj_path)
    train_edge_index, train_edge_weight = get_edge_index_and_weights(train_adj_mat)

    train_data = StaticGraphTemporalSignal(edge_index = train_edge_index, edge_weight = train_edge_weight, features = X_train, targets = Y_train)

    ################ Training ################

    # Create model and optimizers
    model = TemporalGNN(node_features=1, periods=p, f=f)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #device = torch.device("cpu")
    #train_dataset.data.to(device)
    n_epoch = 200

    model = training(model, train_data, n_epoch, optimizer, 128)

    torch.save(model, "aib222687_task2.model")
    #plot_train_val_loss(train_loss_list, val_loss_list, n_epoch)

    ################ Evaluation ################
    #_, train_mae = predict_output(model, data_list, train_mask)
    #_, val_mae = predict_output(model, data_list, val_mask)
    #output, test_mae = predict_output(model, data_list, test_mask)

    #np.savetxt("output_d1.csv", output, fmt="%f", delimiter = ",")
    #print("Train MAE: {:.3f}".format(train_mae))
    #print("Validation MAE: {:.3f}".format(val_mae))
    #print("Test MAE: {:.3f}".format(test_mae))

elif(task == "test"):
    p = int(sys.argv[2])
    f = int(sys.argv[3])
    test_data_path = sys.argv[4]
    output_path = sys.argv[5]
    model_path = sys.argv[6]
    adj_path = "adj_task2_saved.csv"
    graph_splits_path = "graph_splits_task2_saved.npz"

    #p = 12
    #f = 12
    #test_data_path = "test.npz"
    #output_path = "task2_d1_output.npz"
    #model_path = "task2_atgcn_3_f12.model"
    #adj_path = "d1_adj_mx.csv"
    #graph_splits_path = "d1_graph_splits.npz"
    
    ### load model ###
    #model = TemporalGNN(node_features=1, periods=12, f=12)
    model = torch.load(model_path, map_location=get_device())
    
    ### get adjacency matrix, edge index, edge weights ###
    adj_mat = get_adjacency(graph_splits_path, adj_path, label="whole")
    edge_index, edge_weight = get_edge_index_and_weights(adj_mat)
    #print(edge_weight)
    ### get X ###
    #X_test_d1, Y_test_d1 = divide_data_into_windows("d1_graph_splits.npz", "d1_X.csv", 12, 12, label="whole")
    #print(X_test_d1.shape)
    test_data = np.load(test_data_path) 
    X_input = test_data['x']
    X = []
  
    for arr in X_input:
      X.append(arr.T.reshape(-1, 1, arr[:].T.shape[1]))
      #Y.append(arr[p:p+f].T)

    X = np.array(X)

    ### get data list ###

    data_list = StaticGraphTemporalSignal(edge_index = edge_index, edge_weight = edge_weight, features = X, targets = X)

    #### get output ####
          
    output = predict_during_test(model, data_list, f)
  
    np.savez(output_path, y=output)
