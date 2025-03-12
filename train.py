import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# 모델 클래스 정의 (기존 코드와 동일해야 함)
class AttentionBlock(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionBlock, self).__init__()
        self.query = torch.nn.Linear(input_dim, hidden_dim)
        self.key = torch.nn.Linear(input_dim, hidden_dim)
        self.value = torch.nn.Linear(input_dim, hidden_dim)
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)  # Scaled Dot-Product Attention
        attn_weights = self.softmax(attn_scores)
        out = torch.matmul(attn_weights, V)  # Compute attention scores

        return out

class AttentionModel(torch.nn.Module):
    def __init__(self, input_dim=16, hidden_dim=63):
        super(AttentionModel, self).__init__()
        self.attention = AttentionBlock(input_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()  # For binary classification
    
    def forward(self, x):
        x = self.attention(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)  # Ensure output between 0 and 1
        return x

# 폴더 내 모든 CSV 파일 읽기
def load_csv_from_folder(folder_path, label_range=None, fixed_label=None):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    data_list = []
    for file in all_files:
        df = pd.read_csv(file).astype(str)
        df = df.iloc[:, 1:].astype(np.float32)  # 첫 번째 열(파일 이름) 제외
        if fixed_label is not None:
            df['label'] = fixed_label  # 고정된 점수
        else:
            df['label'] = np.random.uniform(label_range[0], label_range[1], size=len(df))  # 범위 내 랜덤 값
        data_list.append(df)
    
    return pd.concat(data_list, ignore_index=True) if data_list else None

# Custom Dataset for Multiple CSVs
class MultiCSVDataset(Dataset):
    def __init__(self, folder_dict):
        data_frames = []
        for folder, label in folder_dict.items():
            if isinstance(label, tuple):
                data = load_csv_from_folder(folder, label_range=label)
            else:
                data = load_csv_from_folder(folder, fixed_label=label)
            if data is not None:
                data_frames.append(data)
        
        if not data_frames:
            raise ValueError("No valid CSV files found in the provided folders.")
        
        full_data = pd.concat(data_frames, ignore_index=True)
        self.X = full_data.iloc[:, :-1].values
        self.y = full_data['label'].values.reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y

# Training Function
def train_model(model, dataloader, epochs=500, lr=0.001, save_path="models_scale"):
    os.makedirs(save_path, exist_ok=True)
    criterion = torch.nn.MSELoss()  # Mean Squared Error Loss for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")
        
        # Save model every 10 epochs
        if (epoch + 1) % (epoch // 5) == 0:
            checkpoint_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved at {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(save_path, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

# Main Execution
if __name__ == "__main__":
    folder_dict = {
        "./1.00": 100,
        "./0.00": 0,
        "./0.10": 10,
        "./0.20": 20,
        "./0.25": 25,
        "./0.30": 30,
        "./0.40": 40
    }
    
    dataset = MultiCSVDataset(folder_dict)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = AttentionModel()
    
    weight_path = "models/final_model_V.pth"  # 변경 가능
    if os.path.exists(weight_path):
        print(f"Loading weights from {weight_path}")
        model.load_state_dict(torch.load(weight_path))
    
    train_model(model, dataloader)
    
    print("Training completed.")
