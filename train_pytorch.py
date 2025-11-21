import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import numpy as np
import argparse
import joblib

# --- CONFIGURACIÓN ---
DATA_CSV = Path("data/lsb_alpha.csv")
MODEL_ONNX_PATH = Path("models_onnx/lsb_alpha.onnx")
INPUT_FEATURES = 63

DATA_SEQ_CSV = Path("data/lsb_seq.csv")
MODEL_SEQ_ONNX_PATH = Path("models_onnx/lsb_seq.onnx")
SEQ_LEN = 20
INPUT_SEQ_FEATURES = INPUT_FEATURES * SEQ_LEN


# --- DEFINICIÓN DEL MODELO (RED NEURONAL) ---
class SignLanguageModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SignLanguageModel, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layer_stack(x)

# --- FUNCIÓN PRINCIPAL DE ENTRENAMIENTO ---
def train_model(model_type, data_path, model_path, input_features):
    print(f"--- Iniciando entrenamiento para el modelo: {model_type} ---")
    
    if not data_path.exists():
        print(f"[ERROR] No se encontró el archivo de datos: {data_path}")
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Asegurando que el directorio '{model_path.parent}' existe.")

    df = pd.read_csv(data_path).dropna()
    print(f"Cargadas {len(df)} muestras desde {data_path}")

    X = df.drop("label", axis=1).values
    y_str = df["label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    
    class_names = list(le.classes_)
    np.save(model_path.with_suffix('.classes.npy'), class_names)
    print(f"Clases detectadas ({len(class_names)}): {class_names}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    joblib.dump(scaler, model_path.with_suffix('.scaler.joblib'))

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = SignLanguageModel(input_size=input_features, num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs_epoch = model(X_test_tensor)
                _, predicted_epoch = torch.max(test_outputs_epoch.data, 1)
                accuracy_epoch = (predicted_epoch == y_test_tensor).sum().item() / len(y_test_tensor)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy_epoch * 100:.2f}%')
            model.train()

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f'\nAccuracy FINAL en el conjunto de prueba: {accuracy * 100:.2f} %')

    dummy_input = torch.randn(1, input_features, requires_grad=True)
    
    torch.onnx.export(model,
                      dummy_input,
                      str(model_path),
                      export_params=True,
                      # <<< CAMBIO FINAL: Eliminamos opset_version para usar el default >>>
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    
    print(f"[SUCCESS] Modelo '{model_type}' entrenado y exportado a {model_path}")
    print("-" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Herramienta de entrenamiento de modelos de señas con PyTorch y ONNX.")
    parser.add_argument('model_type', choices=['static', 'dynamic'], help="El tipo de modelo a entrenar ('static' o 'dynamic').")
    args = parser.parse_args()

    if args.model_type == 'static':
        train_model("Estático", DATA_CSV, MODEL_ONNX_PATH, INPUT_FEATURES)
    elif args.model_type == 'dynamic':
        train_model("Dinámico", DATA_SEQ_CSV, MODEL_SEQ_ONNX_PATH, INPUT_SEQ_FEATURES)