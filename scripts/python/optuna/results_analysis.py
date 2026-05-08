import csv
import os

import numpy as np
import matplotlib.pyplot as plt
import optuna


def load_study_data(db_path, study_name):
    study = optuna.load_study(study_name=study_name, storage=db_path)
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if not trials:
        raise ValueError("No completed trials found in the database.")

    param_names = sorted(trials[0].params.keys())
    data = []
    curves = {}

    for t in trials:
        row = [t.number, t.value] + [t.params[name] for name in param_names]
        data.append(row)
        if t.intermediate_values:
            epochs = sorted(t.intermediate_values.keys())
            curves[t.number] = np.array([[e, t.intermediate_values[e]] for e in epochs])

    return np.array(data), param_names, curves, study.best_trial.number


def create_csv_report(data, param_names, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    header = ["trial_number", "best_value"] + param_names
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    print(f"Successfully saved CSV to: {output_path}")


def plot_from_results(csv_path, curves, best_trial_num, plot_dir):
    """3) Plot from data: Uses the CSV and curve dict to generate visuals."""
    os.makedirs(plot_dir, exist_ok=True)
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    trial_nums = data[:, 0]
    values = data[:, 1]

    #Optimization History
    plt.figure(figsize=(10, 5))
    running_min = np.minimum.accumulate(values)
    plt.scatter(trial_nums, values, color='royalblue', alpha=0.6, label='Trial Result')
    plt.plot(trial_nums, running_min, color='crimson', linestyle='--', label='Best So Far')
    plt.yscale('log')
    plt.xlabel('Trial Number')
    plt.ylabel('Validation Loss')
    plt.title('Hyperparameter Search Progress')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "history.png"))
    plt.close()

    #Loss Curves
    plt.figure(figsize=(10, 6))
    for t_num, curve_data in curves.items():
        alpha = 1.0 if t_num == best_trial_num else 0.2
        color = 'green' if t_num == best_trial_num else 'gray'
        zorder = 5 if t_num == best_trial_num else 1
        
        plt.plot(curve_data[:, 0], curve_data[:, 1], color=color, alpha=alpha, zorder=zorder)

    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Curves (Best Trial in Green)')
    plt.savefig(os.path.join(plot_dir, "curves.png"))
    plt.close()
    print(f"Saved plots to: {plot_dir}")


if __name__ == "__main__":
    db_path = "sqlite:///results/optuna/optuna_study.db"
    study_name = "unet_tuning"
    csv_path = "results/optuna/summary.csv"
    plot_path = "results/optuna/plots"


    study_matrix, params, loss_curves, best_id = load_study_data(db_path, study_name)
    create_csv_report(study_matrix, params, csv_path)     
    plot_from_results(csv_path, loss_curves, best_id, plot_path)