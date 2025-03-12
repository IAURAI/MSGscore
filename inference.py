import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# 모델 클래스 정의 (기존 코드와 동일해야 함)
class AttentionBlock(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=63):
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
        out = torch.matmul(attn_weights, V)
        return out

class AttentionModel(torch.nn.Module):
    def __init__(self, input_dim=16, hidden_dim=63):
        super(AttentionModel, self).__init__()
        self.attention = AttentionBlock(input_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.attention(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 가중치 로드 및 Inference 수행
def load_model_and_infer(model_path, csv_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found!")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} not found!")
    
    # CSV 파일 로드
    data = pd.read_csv(csv_path)
    feature_data = data.iloc[:, 1:].astype(np.float32)  # 첫 번째 열(파일 이름) 제외
    
    input_dim = feature_data.shape[1]
    model = AttentionModel(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 평가 모드 전환
    
    results = []
    with torch.no_grad():
        for _, row in feature_data.iterrows():
            input_tensor = torch.tensor(row.values, dtype=torch.float32).unsqueeze(0)  # Ensure batch dimension
            output = model(input_tensor).item()
            results.append(output)
    
    data['Inference_Result'] = results  # 결과 추가
    return data

# 실행 예시
if __name__ == "__main__":
    model_path = "models/final_model_V7.pth"  # 사용하고 싶은 weight 지정
    csv_path = "./Result_Inference_Mov.csv"  # Inference할 CSV 파일 경로
    
    result_df = load_model_and_infer(model_path, csv_path)
    print(result_df)  # 결과 출력
    
    # 결과를 CSV로 저장
    result_df.to_csv("./inference_results.csv", index=False)
    print("Inference results saved to inference_results.csv")
