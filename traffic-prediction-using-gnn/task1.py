#!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
#!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
#!pip install -q torch-cluster -f https://pytorch-geometric.com/whl/torch-1.12.0+cu113.html
#!pip install -q torch-geometric

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
from torch_geometric.nn import GCNConv, GraphConv, BatchNorm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
from datetime import datetime

def get_np_matrix(df_X_data):
  return df_X_data.iloc[:, :].to_numpy()

def get_train_val_test_nodes(split_path, data_x_path, node_id_list):

  """
  splits = np.load(split_path)

  train_node_ids = splits["train_node_ids"]
  val_node_ids = splits["val_node_ids"]
  test_node_ids = splits["test_node_ids"]
  """
  df_data=pd.read_csv(data_x_path)

  df_x = df_data.iloc[:-1, :]
  df_y = df_data.iloc[1:, :]

  #df_train_x = df_x[train_node_ids.astype(str)]
  #df_train_y = df_y[train_node_ids.astype(str)]

  #df_val_x = df_x[val_node_ids.astype(str)]
  #df_val_y = df_y[val_node_ids.astype(str)]

  #df_test_x = df_x[test_node_ids.astype(str)]
  #df_test_y = df_y[test_node_ids.astype(str)]

  df_x = df_x[node_id_list.astype(str)]
  df_y = df_y[node_id_list.astype(str)]

  X = get_np_matrix(df_x)
  Y = get_np_matrix(df_y)

  """
  X_train = get_np_matrix(df_train_x)
  Y_train = get_np_matrix(df_train_y)

  X_train = get_np_matrix(df_train_x)
  Y_train = get_np_matrix(df_train_y)

  X_val = get_np_matrix(df_val_x)
  Y_val = get_np_matrix(df_val_y)

  X_test = get_np_matrix(df_test_x)
  Y_test = get_np_matrix(df_test_y)
  """

  return  X, Y

def get_adjacency(adj_path):
  df_adj=pd.read_csv(adj_path)
  df_adj.to_csv("adj_task1_saved.csv", index=False)
  
  df_adj=pd.read_csv(adj_path)
  """
  splits = np.load(split_path)

  if label=="train":
    node_ids = splits["train_node_ids"]
  elif label=="val":
    node_ids = splits["val_node_ids"]
  elif label=="test":
    node_ids = splits["test_node_ids"]
  """
  df2 = df_adj.set_index(df_adj.columns[0])

  #print(df2.columns)
  #print(np.array(df2.columns, dtype='int64'))
  return df2.to_numpy(), np.array(df2.columns, dtype='int64')

def get_edge_index_and_weights(adj_matrix):
  #print(adj_matrix.shape)
  adj_matrix_tensor = torch.tensor(adj_matrix, dtype = torch.float)
  edge_index, edge_weight = dense_to_sparse(adj_matrix_tensor)
  return edge_index, edge_weight

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
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels_1):
        super().__init__()
        torch.manual_seed(42)
        self.conv1 = GraphConv(1, hidden_channels_1, aggr="add")
        self.norm = BatchNorm(in_channels=hidden_channels_1)
        self.conv2 = GraphConv(hidden_channels_1, 16, aggr="add")
        self.linear = torch.nn.Linear(16, 1)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.norm(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.linear(x)
        return x


def training(model, data, n_epoch, optimizer, batch_size, train_mask):
    device = get_device()
    
    model.to(device)
    #train_data.data.to(device)
    #val_data.data.to(device)
    
    train_loss_list = []
    #val_loss_list = []

    train_mask = torch.BoolTensor(train_mask).unsqueeze(1).to(device)
    data_loader = DataLoader(data, batch_size=batch_size)
    #val_data_loader = DataLoader(val_data, batch_size=batch_size)
    model.train()
    
    for epoch in tqdm(range(n_epoch)):
        loss_net = 0
        it = 0

        for mini_batch_data in data_loader:
          mini_batch_data = mini_batch_data.to(device)

          out = model(mini_batch_data.x.float(), mini_batch_data.edge_index, mini_batch_data.edge_attr)
          #print("OUT: ", out.shape)
          loss = torch.mean(torch.abs(out[(mini_batch_data.train_mask).flatten()] - mini_batch_data.y[(mini_batch_data.train_mask).flatten()].float()))
          #print("MASKED OUT: ", out[(mini_batch_data.train_mask).flatten()].shape)
          #val_out = model(val_data.data.x.float(), val_data.edge_index.to(device), val_data.edge_weight.to(device))
          #loss_val = torch.mean(torch.abs(val_out - val_data.data.y.float()))

          train_loss_list.append(loss.item())
          #val_loss_list.append(loss_val.item())

          loss.backward()

          loss_net += loss.item()
          it+=1

          optimizer.step()
          optimizer.zero_grad()

        if(epoch%50 == 0 or epoch==n_epoch-1):
            print("\nEpoch {}: Train MAE= {:.4f};".format(epoch+1, loss_net/it))
        
      
    return model, train_loss_list

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

def predict_output(model, data_list, test_mask):
        device = get_device()
        #model = GNN(hidden_channels_1=16)
        #model = torch.load("task1_d1_graphconv_2_layers.model")
        model.to(device)

        output=[]
        test_mae = 0

        for data in data_list:
          data.to(device)
          model.eval()
          out = model(data.x.float(), data.edge_index.to(device), data.edge_attr.to(device))
          #print(out.shape)
          output.append(out.cpu().detach().numpy().reshape(-1,))
          #print(out.shape)
          #loss = criterion()
          #loss = loss(out, train_dataset.data.y.float())
          loss = torch.mean(torch.abs(out[test_mask] - data.y[test_mask].float()))
          test_mae = test_mae + loss.item() 

        

          #out = out.detach().numpy()
        #for i in range(dataset.slices['y'].shape[0]-1):
            #output.append(out[dataset.slices['y'][i]:dataset.slices['y'][i+1], :].cpu().detach().numpy().reshape(-1,))
  
        return np.array(output), test_mae/len(data_list)

def predict_during_test(model, data_list):
  device = get_device()
  #model = GNN(hidden_channels_1=16)
  #model = torch.load("task1_d1_graphconv_2_layers.model")
  model.to(device)
        
  output=[]

  for data in data_list:
      data.to(device)
      model.eval()
      out = model(data.x.float(), data.edge_index.to(device), data.edge_attr.to(device))
      #print(out.shape)
      output.append(out.cpu().detach().numpy().reshape(-1,))
  
  return np.array(output)

  
############################# MAIN ###############################################
task = sys.argv[1]

if(task == "train"):
    train_data_path = sys.argv[2]
    adj_path = sys.argv[3]
    graph_splits_path = sys.argv[4]

    ############# load data ####################
    adj_mat, node_id_list = get_adjacency(adj_path)
    X, Y = get_train_val_test_nodes(graph_splits_path, train_data_path, node_id_list)

    adj_mat, node_id_list = get_adjacency(adj_path)
    edge_index, edge_weight = get_edge_index_and_weights(adj_mat)

    train_mask = get_mask(graph_splits_path, node_id_list, label="train")
    val_mask = get_mask(graph_splits_path, node_id_list, label="val")
    test_mask = get_mask(graph_splits_path, node_id_list, label="test")

    data_list = create_data(X, Y, edge_index, edge_weight, train_mask, val_mask, test_mask)
    #print(len(data_list))

    ################ Training ################

    # Create model and optimizers
    model = GNN(hidden_channels_1=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #device = torch.device("cpu")
    #train_dataset.data.to(device)
    n_epoch = 300

    model, train_loss_list = training(model, data_list, n_epoch, optimizer, 128, train_mask)

    torch.save(model, "aib222687_task1.model")
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
    test_data_path = sys.argv[2]
    output_path = sys.argv[3]
    model_path = sys.argv[4]
    adj_path = "adj_task1_saved.csv"
    
    ### load model ###
    model = GNN(hidden_channels_1=32)
    model = torch.load(model_path, map_location=get_device())
    
    ### get adjacency matrix, edge index, edge weights ###
    adj_mat, node_id_list = get_adjacency(adj_path)
    edge_index, edge_weight = get_edge_index_and_weights(adj_mat)
    
    ### get X ###
    df_init=pd.read_csv(test_data_path)
    df_data = df_init.set_index(df_init.columns[0])

    df_x = df_data.iloc[:, :]
    df_y = df_data.iloc[:, :]

    df_x = df_x[node_id_list.astype(str)]
    df_y = df_y[node_id_list.astype(str)]

    X = get_np_matrix(df_x)
    Y = get_np_matrix(df_y)
  
    ### get data list ###
    data_list = []

    for i in range(X.shape[0]):
          node_features = torch.LongTensor(X[i].reshape(-1,1))

          x = node_features
          y = torch.LongTensor(Y[i].reshape(-1,1))

          data = Data(x=x, edge_index = edge_index, edge_attr = edge_weight.unsqueeze(1), y=y)
          data_list.append(data)
          
    output = predict_during_test(model, data_list)
    
    timestamps = list(df_data.index)
    new_timestamps = timestamps[1:]
    new_timestamps.append(str(datetime.strptime(df_data.index[-1], '%Y-%m-%d %H:%M:%S') - datetime.strptime('00:00:00', '%H:%M:%S') + datetime.strptime('00:05:00', '%H:%M:%S')))
    column_name_list = list(df_data.columns)
    
    df_output = pd.DataFrame(data=output, index = new_timestamps, columns = column_name_list)
    
    #print(df_output)
    df_output.to_csv(output_path)
