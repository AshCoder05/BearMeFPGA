import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, classification_report

# 0: Healthy, 1: Ball, 2: Inner Race, 3: Outer Race
CWRU_FILES = {
    0: "97.mat",
    1: "118.mat",
    2: "105.mat",
    3: "130.mat"
}

# ==========================================
# 1. HARDWARE-ACCURATE FEATURE PIPELINE
# ==========================================
# Pre-compute Hann window (This will be a ROM in Verilog)
HANN_WINDOW = np.hanning(1024)

def extract_features(raw_signal):
    """Raw -> Abs -> Hann Window -> FFT -> Drop DC -> Mag^2 -> 16 Band Energy"""
    # 1. Rectify (Absolute Value)
    rectified = np.abs(raw_signal)
    
    # 2. Apply Window (Prevents spectral leakage)
    windowed = rectified * HANN_WINDOW
    
    # 3. FFT
    fft_vals = np.fft.fft(windowed)
    
    # 4. Mag^2 (Skip Bin 0 to block DC offset, take next 512 bins)
    # Bins 1 to 512 represent the usable AC spectrum
    mag_sq = np.real(fft_vals[1:513])**2 + np.imag(fft_vals[1:513])**2
    
    # 5. 16 Band Energy (512 bins / 16 = 32 bins per band)
    energies = np.zeros(16)
    for i in range(16):
        energies[i] = np.sum(mag_sq[i*32 : (i+1)*32])
        
    return energies

# ==========================================
# 2. STRICT DATA LOADING (NO LEAKAGE)
# ==========================================
def window_signal(raw_array, class_label):
    """Applies sliding window to a 1D array."""
    window_size = 1024
    stride = 512
    features, labels = [], []
    for start in range(0, len(raw_array) - window_size, stride):
        window = raw_array[start : start + window_size]
        features.append(extract_features(window))
        labels.append(class_label)
    return features, labels

def get_train_test_data():
    """Splits raw data FIRST, then windows it to prevent leakage."""
    print("--- Loading & Splitting CWRU Dataset ---")
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    for label, filename in CWRU_FILES.items():
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Missing {filename}.")
            
        data = loadmat(filename)
        de_key = next(key for key in data.keys() if 'DE_time' in key)
        raw_signal = data[de_key].flatten()[:50000] # Take first 50k samples
        
        # SPLIT RAW SIGNAL FIRST (70% Train, 30% Test)
        split_idx = int(len(raw_signal) * 0.7)
        train_raw = raw_signal[:split_idx]
        test_raw = raw_signal[split_idx:]
        
        # Window the splits independently
        tr_f, tr_l = window_signal(train_raw, label)
        te_f, te_l = window_signal(test_raw, label)
        
        X_train_list.extend(tr_f)
        y_train_list.extend(tr_l)
        X_test_list.extend(te_f)
        y_test_list.extend(te_l)
        
        print(f"Class {label}: {len(tr_f)} Train windows, {len(te_f)} Test windows.")
        
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)
    
    # Shuffle Training Data only
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    # SCALE BASED ON TRAIN SET ONLY (Prevent info leak)
    global_max = np.max(X_train)
    X_train_scaled = (X_train / global_max) * 3.0 
    X_test_scaled = (X_test / global_max) * 3.0
    
    return (torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long),
            torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

# ==========================================
# 3. MODEL & TRAINING
# ==========================================
class SmallMLP(nn.Module):
    def __init__(self):
        super(SmallMLP, self).__init__()
        self.fc1 = nn.Linear(16, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def train_model(X_train, y_train):
    model = SmallMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print("\n--- Training MLP on Strict Train Set ---")
    for epoch in range(500):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 100 == 0:
            _, preds = torch.max(out, 1)
            acc = (preds == y_train).float().mean().item() * 100
            print(f"Epoch {epoch+1}/500 | Loss: {loss.item():.4f} | Train Acc: {acc:.2f}%")
            
    return model

# ==========================================
# 4. HARDWARE QUANTIZATION & INFERENCE
# ==========================================
SCALE_FACTOR = 2**13

def quantize_q2_13(tensor_or_array):
    if isinstance(tensor_or_array, torch.Tensor):
        arr = tensor_or_array.detach().numpy()
    else:
        arr = tensor_or_array
    quantized = np.clip(np.round(arr * SCALE_FACTOR), -32768, 32767)
    return quantized.astype(np.int16)

def fixed_inference(x_q, W1_q, b1_q, W2_q, b2_q):
    z1_mac = np.dot(x_q.astype(np.int64), W1_q.T.astype(np.int64)) 
    z1 = (z1_mac >> 13) + b1_q.astype(np.int64)
    z1 = np.clip(np.maximum(z1, 0), -32768, 32767).astype(np.int16)

    z2_mac = np.dot(z1.astype(np.int64), W2_q.T.astype(np.int64))
    z2 = (z2_mac >> 13) + b2_q.astype(np.int64)
    return np.argmax(z2)

def export_hex(filename, data_q):
    with open(filename, "w") as f:
        for val in data_q.flatten():
            f.write(f"{int(val) & 0xFFFF:04x}\n")

# ==========================================
# 5. EXECUTE, EXPORT & EVALUATE
# ==========================================
def run_export_pipeline():
    os.makedirs("test_vectors", exist_ok=True)
    
    # 1. Get isolated Train/Test splits
    X_train, y_train, X_test, y_test = get_train_test_data()
    
    # 2. Train Model
    model = train_model(X_train, y_train)
    
    # 3. Quantize Weights
    W1_q = quantize_q2_13(model.fc1.weight)
    b1_q = quantize_q2_13(model.fc1.bias)
    W2_q = quantize_q2_13(model.fc2.weight)
    b2_q = quantize_q2_13(model.fc2.bias)
    
    # 4. Export Hex Files
    print("\n--- Exporting Hardware Files ---")
    export_hex("weights_l1.hex", W1_q)
    export_hex("bias_l1.hex", b1_q)
    export_hex("weights_l2.hex", W2_q)
    export_hex("bias_l2.hex", b2_q)
    
    # 5. Evaluate on STRICT TEST SET
    print("\n--- Evaluating on Test Set (Unseen Data) ---")
    y_true_list = []
    y_pred_fixed_list = []
    
    float_correct = 0
    fixed_correct = 0
    total_test = len(X_test)
    
    for i in range(total_test):
        x = X_test[i].numpy()
        y_true = y_test[i].item()
        y_true_list.append(y_true)
        
        # Float Inference
        out_float = model(torch.tensor(x, dtype=torch.float32)).detach().numpy()
        pred_float = np.argmax(out_float)
        if pred_float == y_true: float_correct += 1
        
        # Fixed Inference
        x_q = quantize_q2_13(x)
        pred_fixed = fixed_inference(x_q, W1_q, b1_q, W2_q, b2_q)
        y_pred_fixed_list.append(pred_fixed)
        if pred_fixed == y_true: fixed_correct += 1
        
        # Export a few test vectors
        if i < 20:
            export_hex(f"test_vectors/sample_{i+1:02d}.hex", x_q)

    # 6. Metrics and Confusion Matrix
    print(f"Float Accuracy: {float_correct/total_test * 100:.2f}%")
    print(f"Fixed Accuracy: {fixed_correct/total_test * 100:.2f}%\n")
    
    print("Confusion Matrix (Fixed Point Model):")
    
    print(confusion_matrix(y_true_list, y_pred_fixed_list))
    
    print("\nPer-Class Classification Report:")
    print(classification_report(y_true_list, y_pred_fixed_list, target_names=['Healthy', 'Ball', 'Inner', 'Outer']))

if __name__ == "__main__":
    run_export_pipeline()
