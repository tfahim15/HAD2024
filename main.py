import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
import random, math
from torchmetrics.classification import AUROC
from sklearn.model_selection import KFold
import argparse

# Argument parser for command line input (dataset, fold number, epochs)
parser = argparse.ArgumentParser(description="Process dataset and fold number.")
parser.add_argument(
        '--dataset', 
        type=str, 
        required=True, 
        help='The name or path of the dataset'
    )
parser.add_argument(
        '--fold_no', 
        type=int, 
        required=True, 
        help='The fold number for cross-validation'
    )
parser.add_argument(
        '--epochs', 
        type=int, 
        required=True, 
        help='The number of maximum epochs'
    )

args = parser.parse_args()


# Load arguments
dataset = args.dataset
fold_no = args.fold_no
epochs = args.epochs


# Load dataset
data = torch.load("data/"+dataset+".pt")


# Extract hypergraphs
H = data["H"]
H_ = data["H_"]


# Perform 5-fold cross-validation
kf = KFold(n_splits=5)
# Store train and test indices for each fold
splits = []
for train_index, test_index in kf.split([i for i in range(H.shape[0])]):
    splits.append((train_index, test_index))


# Select train and test indices for the specified fold
train_index, test_index = splits[fold_no]


# Partition hypergraph matrix for training and testing
H, H_pos_test = H[train_index], H[test_index]


# Adjust the size of the test set to match the negative samples
m = math.ceil(len(H_)/len(H_pos_test))
H_pos_test = torch.cat([H_pos_test for _ in range(m)])[:len(H_)]

# Load node features
node_feat = data["node_feat"]


# Define a neural network (HNN)
class HNN(torch.nn.Module):
    def __init__(self, inp, outp):
        super(HNN, self).__init__()

        #out: 1 x emdedding_dim
        self.linear1 = nn.Linear(inp, outp)
        self.activation_function1 = nn.ReLU()


    def forward(self, inputs):
        out = self.linear1(inputs)
        out = self.activation_function1(out)
        return out

# Hyperparameters
EMDEDDING_DIM = 128
n_features = data["node_feat"].shape[1]


# Initialize neural network models
VNN0 = HNN(n_features, EMDEDDING_DIM)
ENN0 = HNN(EMDEDDING_DIM,8)
VNN1 = HNN(8,8)
ENN1 = HNN(8,8)

# Activation function
activation = torch.sqrt

# Optimizer
optimizer = torch.optim.Adam([*VNN0.parameters(), *ENN0.parameters(), *VNN1.parameters(), *ENN1.parameters()], lr=0.0001)

# Initialize AUROC metric and training loop variables
c = None
auroc = AUROC(task="binary")
ep = 0

# Training loop
for i in tqdm(range(epochs)):
    total_loss = 0
    E = ENN0(torch.matmul(H, VNN0(node_feat)))
    V = VNN1(torch.matmul(torch.transpose(H, 0, 1), E))

    E_new = torch.zeros_like(E)
    for i in range(len(E_new)):
        max_vals,_ = torch.max(V[H[i]==1], dim=0)
        min_vals,_ = torch.min(V[H[i]==1], dim=0)
        E_new[i] = max_vals - min_vals

    E = E_new
    c = torch.mean(E, dim = 0)
    
    probs = torch.sum((E-c)**2, dim=1)
    probs = activation(probs)
    cost = torch.mean(probs) 
    print(cost)
    
    # Early stopping condition
    if float(cost)<0.0001 or torch.isnan(cost):
        break
        
    # Backpropagation and optimizer step
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

# Evaluation mode for networks after training
ENN0.eval()
VNN0.eval()
ENN1.eval()
VNN1.eval()

# Compute anomaly scores for the test set
E = ENN0(torch.matmul(H_pos_test, VNN0(node_feat)))
V = VNN1(torch.matmul(torch.transpose(H_pos_test, 0, 1), E))
E_new = torch.zeros_like(E)
for i in range(len(E_new)):
    max_vals,_ = torch.max(V[H_pos_test[i]==1], dim=0)
    min_vals,_ = torch.min(V[H_pos_test[i]==1], dim=0)
    E_new[i] = max_vals - min_vals

E = E_new
probs = torch.sum((E-c)**2, dim=1)
probs = activation(probs)
pos_prob = probs
cost = torch.mean(probs)
print(cost)


# Compute anomaly scores for the test set
E = ENN0(torch.matmul(H_, VNN0(node_feat)))
V = VNN1(torch.matmul(torch.transpose(H_, 0, 1), E))
E_new = torch.zeros_like(E)
for i in range(len(E_new)):
    max_vals,_ = torch.max(V[H_[i]==1], dim=0)
    min_vals,_ = torch.min(V[H_[i]==1], dim=0)
    E_new[i] = max_vals - min_vals

E = E_new
probs = torch.sum((E-c)**2, dim=1)
probs = activation(probs)
neg_prob = probs
cost = torch.mean(probs)
print(cost)


# Concatenate positive and negative probabilities and compute AUROC
probs = torch.cat((pos_prob, neg_prob))
pos_labels = torch.tensor([0 for i in range(len(pos_prob))])
neg_labels = torch.tensor([1 for i in range(len(neg_prob))])
labels = torch.cat((pos_labels, neg_labels))

# Calculate and print AUROC score
print("AUROC:", float(auroc(probs, labels)))

