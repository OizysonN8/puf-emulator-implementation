"""
Deep Learning Attack on PUF Challenge–Response Data

Reads our existing challenge/response pairs csv files and trains:
- Logistic Regression baseline (optional)
- PyTorch MLP attacker model

Outputs:
- test accuracy
- confusion matrix
- TensorBoard logs
- best-model checkpoint
- Results saved as CSV
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import random

# -------------------------------------------------------------------
# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Arbiter PUF Feature Map φ(C)
def phi_feature_map(chal):
    """
    chal: numpy array (M, n) of 0/1 bits
    return: φ(C) (M, n) float32
    """
    X = 1 - 2 * chal  # {0,1} -> {+1,-1}
    M, n = X.shape
    Phi = np.empty((M, n), dtype=np.float32)
    running = np.ones(M)
    for j in range(n - 1, -1, -1):
        running *= X[:, j]
        Phi[:, j] = running
    return Phi
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Dataset
class CRPDataset(Dataset):
    """
    Loads CRPs from a CSV file.
    Supports two formats:
        1. "Challenge" = bitstring, "Response" = 0/1
        2. challenge_0, challenge_1, ..., challenge_n + response column

    Automatically:
        - Detects correct response column
        - Detects format
        - Fixes malformed bitstrings
        - Pads bitstrings to max length
    """

    def __init__(self, csv_path, feature_map="phi"):
        df = pd.read_csv(csv_path)

        # ------------------------------
        # Detect RESPONSE column
        # ------------------------------
        resp_candidates = ["response", "Response", "RESP", "label", "Label"]
        resp_col = None
        for c in resp_candidates:
            if c in df.columns:
                resp_col = c
                break
        if resp_col is None:
            raise ValueError("CSV must contain a Response/response column.")

        # ------------------------------
        # Detect CHALLENGE format
        # ------------------------------
        if "Challenge" in df.columns:
            # CASE 1: bitstring column
            bitstrings = df["Challenge"].astype(str).tolist()

            # compute bit lengths
            lengths = [len(s) for s in bitstrings]
            n_bits = max(lengths)

            # warn & pad malformed rows
            bad = sum(1 for s in bitstrings if len(s) != n_bits)
            if bad > 0:
                print(f"[WARN] {bad} CRPs have inconsistent bitstring length. Padding applied.")

            # pad shorter bitstrings
            fixed = []
            for s in bitstrings:
                if len(s) < n_bits:
                    s = s.ljust(n_bits, "0")
                fixed.append(s)

            # convert to numpy matrix
            X_raw = np.zeros((len(fixed), n_bits), dtype=np.int8)
            for i, s in enumerate(fixed):
                X_raw[i] = np.fromiter((int(b) for b in s), count=n_bits, dtype=np.int8)

        else:
            # CASE 2: explicit challenge_0...challenge_n columns
            chal_cols = [c for c in df.columns if c.lower().startswith("challenge_")]
            if len(chal_cols) == 0:
                raise ValueError("CSV must have a 'Challenge' bitstring or challenge_# columns.")
            X_raw = df[chal_cols].astype(np.int8).to_numpy()

        # ------------------------------
        # RESPONSE vector
        # ------------------------------
        y = df[resp_col].astype(np.int8).to_numpy()

        # ------------------------------
        # Apply feature map
        # ------------------------------
        if feature_map == "phi":
            X = phi_feature_map(X_raw)
        else:
            X = X_raw.astype(np.float32)

        # ------------------------------
        # Convert to torch tensors
        # ------------------------------
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# MLP Attack Model
class MLP(nn.Module):
    def __init__(self, input_dim, h1=256, h2=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 2)
        )
    def forward(self, x): return self.net(x)
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Training helpers
def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    losses, preds, trues = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        trues.extend(yb.cpu().numpy())
    return np.mean(losses), accuracy_score(trues, preds)

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    losses, preds, trues = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            losses.append(loss_fn(logits, yb).item())
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(yb.cpu().numpy())
    return np.mean(losses), accuracy_score(trues, preds), confusion_matrix(trues, preds)
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Main program
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)

    # Load dataset
    ds = CRPDataset(args.csv, feature_map=args.feature_map)
    N = len(ds)

    # Train/val/test split
    test_n = int(0.15 * N)
    val_n = int(0.15 * N)
    train_n = N - test_n - val_n
    train_ds, val_ds, test_ds = random_split(
        ds, [train_n, val_n, test_n],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch)
    test_loader  = DataLoader(test_ds, batch_size=args.batch)

    # TensorBoard logger
    run_id = datetime.now().strftime("%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(args.log_dir, f"dl_attack_{run_id}"))

    # Logistic Regression Baseline
    if args.baseline:
        print("[LOGREG] Training baseline logistic regression model…")
        X = ds.X.numpy()
        y = ds.y.numpy()
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[:train_n], y[:train_n])
        pred = clf.predict(X[train_n:])
        acc = accuracy_score(y[train_n:], pred)
        cm = confusion_matrix(y[train_n:], pred)
        
        print(f"[LOGREG] Baseline accuracy: {acc:.4f}")
        print("Confusion Matrix:\n", cm)
        
        # Save baseline results to CSV
        results_df = pd.DataFrame({
            'run_id': [run_id],
            'model': ['LogisticRegression'],
            'dataset': [args.csv],
            'feature_map': [args.feature_map],
            'n_samples': [N],
            'n_train': [train_n],
            'n_val': [val_n],
            'n_test': [test_n - val_n],
            'test_accuracy': [acc],
            'tn': [cm[0, 0]],
            'fp': [cm[0, 1]],
            'fn': [cm[1, 0]],
            'tp': [cm[1, 1]],
            'epochs': [None],
            'batch_size': [None],
            'lr': [None],
            'h1': [None],
            'h2': [None],
            'dropout': [None]
        })
        
        out_csv = os.path.join(args.log_dir, f"results_{run_id}.csv")
        results_df.to_csv(out_csv, index=False)
        print(f"\nSaved results to {out_csv}")
        
        # Also append to master results file
        master_csv = os.path.join(args.log_dir, "all_results.csv")
        if os.path.exists(master_csv):
            results_df.to_csv(master_csv, mode='a', header=False, index=False)
        else:
            results_df.to_csv(master_csv, index=False)
        print(f"Appended to {master_csv}")
        
        return

    # MLP Attack Model
    input_dim = ds.X.shape[1]
    model = MLP(input_dim, args.h1, args.h2, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0
    best_state = None

    # Training loop
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, loss_fn, device)
        va_loss, va_acc, _ = eval_epoch(model, val_loader, loss_fn, device)

        writer.add_scalar("loss/train", tr_loss, ep)
        writer.add_scalar("loss/val", va_loss, ep)
        writer.add_scalar("acc/train", tr_acc, ep)
        writer.add_scalar("acc/val", va_acc, ep)

        print(f"Epoch {ep:03d} | Train Acc: {tr_acc:.4f} | Val Acc: {va_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            best_state = model.state_dict().copy()

    # Load best model
    model.load_state_dict(best_state)

    # Final test
    te_loss, te_acc, te_cm = eval_epoch(model, test_loader, loss_fn, device)
    print(f"\nFINAL TEST ACCURACY: {te_acc:.4f}")
    print("Confusion Matrix:\n", te_cm)

    # Save results to CSV
    results_df = pd.DataFrame({
        'run_id': [run_id],
        'model': ['MLP'],
        'dataset': [args.csv],
        'feature_map': [args.feature_map],
        'n_samples': [N],
        'n_train': [train_n],
        'n_val': [val_n],
        'n_test': [test_n],
        'test_accuracy': [te_acc],
        'test_loss': [te_loss],
        'best_val_accuracy': [best_val],
        'tn': [te_cm[0, 0]],
        'fp': [te_cm[0, 1]],
        'fn': [te_cm[1, 0]],
        'tp': [te_cm[1, 1]],
        'epochs': [args.epochs],
        'batch_size': [args.batch],
        'lr': [args.lr],
        'h1': [args.h1],
        'h2': [args.h2],
        'dropout': [args.dropout]
    })
    
    out_csv = os.path.join(args.log_dir, f"results_{run_id}.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")
    
    # Also append to master results file for easy comparison
    master_csv = os.path.join(args.log_dir, "all_results.csv")
    if os.path.exists(master_csv):
        results_df.to_csv(master_csv, mode='a', header=False, index=False)
    else:
        results_df.to_csv(master_csv, index=False)
    print(f"Appended to {master_csv}")
    
    writer.close()
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="ArbiterPUF\\arbiter_crp.csv")
    parser.add_argument("--feature_map", type=str, default="phi", choices=["phi","raw"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--h1", type=int, default=256)
    parser.add_argument("--h2", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--log_dir", type=str, default="MultilayerPerceptronModel/results")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)