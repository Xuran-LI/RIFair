import os
import numpy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ACSEmployment_test.model import MLP, WideDeep, DeepFM, AutoInt, TabTransformer


# ============================================================
# Dataset wrapper
# ============================================================
class TabularDataset(Dataset):
    def __init__(self, index_data, value_data, labels):
        """
        index_data: [N, num_fields] int64
        value_data: [N, num_fields] float32
        labels: [N, num_classes] one-hot float32
        """
        self.index_data = torch.tensor(index_data, dtype=torch.long) if index_data is not None else None
        self.value_data = torch.tensor(value_data, dtype=torch.float32) if value_data is not None else None
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.index_data is not None and self.value_data is not None:
            return self.index_data[idx], self.value_data[idx], self.labels[idx]
        else:
            return self.value_data[idx], self.labels[idx]


# ============================================================
# Unified training function
# ============================================================
def train_model(model, dataset, name, output_dir="../dataset/ACS/employment/model/", batch_size=64, epochs=25, lr=1e-3,
                device='cpu'):
    """
    training for all tabular models (MLP, WideDeep, DeepFM, AutoInt, TabTransformer)
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{name}.pt")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        for batch in dataloader:
            if len(batch) == 3:
                x_index, x_value, y = batch
                x_index, x_value, y = x_index.to(device), x_value.to(device), y.to(device)
                outputs = model(x_index, x_value)
            else:
                x_value, y = batch
                x_value, y = x_value.to(device), y.to(device)
                outputs = model(x_value)

            loss = criterion(outputs, y.argmax(dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_acc += (outputs.argmax(dim=1) == y.argmax(dim=1)).sum().item()

        epoch_loss = total_loss / len(dataset)
        epoch_acc = total_acc / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"✔ Saved {name} → {save_path}")
    return model


# ============================================================
# Example: training all models
# ============================================================
if __name__ == "__main__":
    # ---------------- Load Keras-style numpy data ----------------
    N_O_train_index = numpy.load("../dataset/ACS/employment/data/N_O_train_i.npy", allow_pickle=True)
    N_O_train_value = numpy.load("../dataset/ACS/employment/data/N_O_train_V.npy", allow_pickle=True)
    N_O_train_label = numpy.load("../dataset/ACS/employment/data/N_O_train_y.npy", allow_pickle=True)
    N_O_train_label = numpy.eye(2)[N_O_train_label]  # one-hot

    N_E_train_index = numpy.load("../dataset/ACS/employment/data/N_E_train_i.npy", allow_pickle=True)
    N_E_train_value = numpy.load("../dataset/ACS/employment/data/N_E_train_V.npy", allow_pickle=True)
    N_E_train_label = numpy.load("../dataset/ACS/employment/data/N_E_train_y.npy", allow_pickle=True)
    N_E_train_label = numpy.eye(2)[N_E_train_label]

    num_fields = N_O_train_index.shape[1]

    # ---------------- Create Datasets ----------------
    dataset_O = TabularDataset(N_O_train_index, N_O_train_value, N_O_train_label)
    dataset_E = TabularDataset(N_E_train_index, N_E_train_value, N_E_train_label)

    # ---------------- Example: MLP ----------------
    mlp_input = N_O_train_index * N_O_train_value  # depends on your preprocessing
    dataset_mlp = TabularDataset(None, mlp_input, N_O_train_label)
    mlp = MLP(input_dim=mlp_input.shape[1])
    train_model(mlp, dataset_mlp, name="MLP")

    max_value = numpy.load("../dataset/ACS/employment/data/max_values.npy")
    min_value = numpy.load("../dataset/ACS/employment/data/min_values.npy")
    num_categories = max_value - min_value + 1

    # ---------------- Wide & Deep ----------------
    wd_model = WideDeep(num_fields=num_fields)
    train_model(wd_model, dataset_E, name="WideDeep")

    # ---------------- DeepFM ----------------
    dfm_model = DeepFM(num_fields=num_fields)
    train_model(dfm_model, dataset_E, name="DeepFM")

    # ---------------- AutoInt ----------------
    autoint_model = AutoInt(vocab_size=100, embed_size=32, field_size=num_fields)
    train_model(autoint_model, dataset_E, name="AutoInt")

    # ---------------- TabTransformer ----------------
    dataset_tab = TabularDataset(None, N_O_train_index * N_O_train_value, N_O_train_label)
    tabtf_model = TabTransformer(num_fields=num_fields)
    train_model(tabtf_model, dataset_tab, name="TabTransformer")
