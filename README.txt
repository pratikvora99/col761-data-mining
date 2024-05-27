Team Members:
Pratik Vora (2022AIB2687) 33%
Anwoy Chatterjee (2022AIB2675) 33%
Dhruvil Sheth (2022AIB2689) 33%

-----------------------------------------

In Task-2, we have implemented a model based on [1]. For the implementation we have used the pytorch geometric temporal library.
The model is an extension to the TGCN model [2], but also uses attention.

It consists of three blocks, each of 2 layers: 'GCNConv' and a 'Linear' layer. As we have used number of hidden channels = 16, so the number of out_channels for each GCNConv layer is 16. 

At the end, another 'Linear' layer is used where number of input features = 16, and the number of output features = f.

The model is trained batchwise, using mini-batch size of 128, and Adam optimizer.


References:

[1] Zhu, J., Song, Y., Zhao, L., & Li, H. (2020). A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting. arXiv. https://doi.org/10.48550/ARXIV.
[2] Zhao, L., Song, Y., Zhang, C., Liu, Y., Wang, P., Lin, T., Deng, M., & Li, H. (2020). T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction. IEEE Transactions on Intelligent Transportation Systems, 21(9), 3848–3858. https://doi.org/10.1109/tits.2019.2935152