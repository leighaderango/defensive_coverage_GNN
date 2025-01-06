import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
import os
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd

os.chdir('..')

class GraphClassifierWithMetadata(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, metadata_dim, dropout_rate = 0.3):
        super(GraphClassifierWithMetadata, self).__init__()
        # graph convolutions
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        
        # fully connected layers for combined graph embedding and metadata
        self.fc1 = torch.nn.Linear(hidden_dim + metadata_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def node_wise_normalization(self, x):
        norm = torch.norm(x, p=2, dim=1, keepdim=True)  # compute L2 norm for each node
        norm = torch.clamp(norm, min=1e-6)  # avoid division by zero
        return x / norm

    def forward(self, x, edge_index, edge_attr, batch, metadata):
        # normalize node and edge features to assist in gradient descent
        x = self.node_wise_normalization(x)
        edge_attr = self.node_wise_normalization(edge_attr)

        # convolve
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)


        # pool node embeddings to get a graph-level embedding
        graph_embedding = global_mean_pool(x, batch)

        # combine graph embedding with metadata
        combined_embedding = torch.cat([graph_embedding, metadata], dim=1)

        # pass through fully connected layers
        out = self.fc1(combined_embedding)
        out = F.relu(out)
        out = self.fc2(out)
        
        return out
    

dataset = torch.load("mofo-mofc/mofoc_graphs.pt")


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
all_loader = DataLoader(dataset, batch_size = 32, shuffle = True)

model = GraphClassifierWithMetadata(input_dim=9, hidden_dim=64, output_dim=3, metadata_dim = 2) 
    # input_dim = num node feaetures
    # hidden_dim = hyperparameter
    # output_dim: number of class responses
    # metadata_dim: number of non-response graph-level features
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


# training loop
for epoch in range(30):
    model.train()
    total_loss = 0
    for data in all_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.y[:, 1:3])
        loss = criterion(out, data.y[:, 0])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


torch.save(model, 'mofo-mofc/mofoc_model.pth')



model = torch.load('mofo-mofc/mofoc_model.pth')

# Evaluation
model.eval()
predictions = []
labels = []
ids = []
probs = []

with torch.no_grad():
    for data in all_loader:
        out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.y[:, 1:3])
        pred = out.argmax(dim=1)
        predictions.extend(pred.cpu().numpy())
        labels.extend(data.y[:, 0].long().cpu().numpy())
        ids.extend(data.y[:, 3:])
        probs.extend(F.softmax(out))

# Compute evaluation metrics
conf_matrix = confusion_matrix(labels, predictions)
accuracy = accuracy_score(labels, predictions)
report = classification_report(labels, predictions)

print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)


gameids = [id[0].item() for id in ids]
playids = [id[1].item() for id in ids]
probclosed = [prob[0].item() for prob in probs]
probopen = [prob[1].item() for prob in probs]
probred = [prob[2].item() for prob in probs]


data = {'gameId': gameids, 
        'playId': playids,
        'preds': predictions, 
        'true': labels, 
        'closed_prob': probclosed,
        'open_prob': probopen,
        'red_prob': probred}

mofoc_preds = pd.DataFrame(data)

