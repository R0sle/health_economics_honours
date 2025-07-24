from sklearn.neighbors import KNeighborsRegressor

import numpy as np
import optuna
import sys
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

X_train = np.load(sys.argv[1])
y_train = np.load(sys.argv[2])
X_test = np.load(sys.argv[3])
y_test = np.load(sys.argv[4])
output_dir = sys.argv[5]

X_train_tensor = torch.from_numpy((X_train).values.astype(np.float32)).float()
y_train_tensor = torch.from_numpy((y_train).values.astype(np.float32)).float()
X_test_tensor = torch.from_numpy((X_test).values.astype(np.float32)).float()
y_test_tensor = torch.from_numpy((y_test).values.astype(np.float32)).float()

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=True)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, num_layers, output_dim, activation):
        super().__init__()
        layers = []
        current_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_units[i]))
            layers.append(activation)
            current_dim = hidden_units[i]
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
#using epsilon to avoid errors caused by division by zero 
def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-8):
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) 

def objective(trial):

    # Suggest values for the hyperparameters using a trial object.
    number_hidden_layers = trial.suggest_int("number_hidden_layers", 1, 10)
    number_hidden_units = [trial.suggest_int(f"n_units_l{i}", 4, 128) for i in range(number_hidden_layers)]

    lr = trial.suggest_float("lr", 1e-4, 1e-2)

    weight_decay = trial.suggest_float("wd", 1e-4, 1e-2)

    scheduler_type = trial.suggest_categorical("scheduler", ["None", "StepLR", "ReduceLROnPlateau"])

    activation_name = trial.suggest_categorical("activation_fn", ['relu', 'tanh'])

    activation_fn = None
    if activation_name == 'relu':
            activation_fn = nn.ReLU()
    else: activation_fn = nn.Tanh()

    model = MLP(input_dim=435, hidden_units=number_hidden_units, num_layers=number_hidden_layers, output_dim=1, activation=activation_fn)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    if scheduler_type == "StepLR":
        #how often the learning rate is decayed
        step_size = trial.suggest_int("step_size", 5, 10)
        #factor by which the learning rate is decayed
        gamma = trial.suggest_float("gamma", 0.1, 0.5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "ReduceLROnPlateau":
        factor = trial.suggest_float("plateau_factor", 0.25, 0.75)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=3)
    else:
        scheduler = None

    # Training loop
    for epoch in range(30):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = mean_absolute_percentage_error(output, yb)
            loss.backward()
            optimizer.step()

        # Validation loss
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                output = model(xb)
                test_loss += mean_absolute_percentage_error(output, yb)

        test_loss /= len(test_loader)

        # Step the scheduler
        if scheduler_type == "ReduceLROnPlateau":
            scheduler.step(test_loss)
        elif scheduler:
            scheduler.step()

    if trial.number == 0 or test_loss < trial.study.best_value:
        torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
        with open(f"{output_dir}/best_params.json", "w") as f:
            json.dump(trial.params, f, indent=2)

    return test_loss  # Optuna minimizes this

#Create a study object and optimize the objective function.
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=300)

# Save study for later visualization
joblib.dump(study, f"{output_dir}/optuna_study.pkl")

import json

dataset_name = sys.argv[6]
imputation_method = sys.argv[7]

summary = {
    "dataset": dataset_name,
    "imputation": imputation_method,
    "model": 'multi-layer perceptron',
    "best_params": study.best_params,
    "validation_accuracy": study.best_value,
}

with open(f"{output_dir}/results.json", "w") as f:
    json.dump(summary, f, indent=2)