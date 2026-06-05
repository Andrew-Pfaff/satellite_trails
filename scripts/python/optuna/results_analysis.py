import csv
import os
import argparse

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


def plot_from_results(csv_path, curves, best_trial_num, plot_dir, model_type):
    """3) Plot from data: Uses the CSV and curve dict to generate visuals."""
    os.makedirs(plot_dir, exist_ok=True)
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    trial_nums = data[:, 0]
    values = data[:, 1]

    #Optimization History
    is_maximise = (model_type == "classifier")
    metric_label = "Penalized Specificity" if is_maximise else "Val IoU"

    plt.figure(figsize=(10, 5))
    if is_maximise:
        running_best = np.maximum.accumulate(values)
    else:
        running_best = np.maximum.accumulate(values)
    plt.scatter(trial_nums, values, color='royalblue', alpha=0.6, label='Trial Result')
    plt.plot(trial_nums, running_best, color='crimson', linestyle='--', label='Best So Far')
    plt.xlabel('Trial Number')
    plt.ylabel(metric_label)
    plt.title('Hyperparameter Search Progress')
    plt.legend()
    plt.ylim(bottom=0.7)
    plt.savefig(os.path.join(plot_dir, "history.png"))
    plt.close()

    #Loss Curves
    plt.figure(figsize=(10, 6))
    for t_num, curve_data in curves.items():
        alpha = 1.0 if t_num == best_trial_num else 0.2
        color = 'green' if t_num == best_trial_num else 'gray'
        zorder = 5 if t_num == best_trial_num else 1
        plt.plot(curve_data[:, 0], curve_data[:, 1], color=color, alpha=alpha, zorder=zorder)
    plt.xlabel('Epoch')
    plt.ylabel(metric_label)
    plt.title(f'Validation Curves (Best Trial in Green)')
    plt.savefig(os.path.join(plot_dir, "curves.png"))
    plt.close()
    print(f"Saved plots to: {plot_dir}")



def parse_args():
    parser = argparse.ArgumentParser(description="Plot Optuna study results")
    parser.add_argument("--model-type", type=str, required=True, choices=["unet", "classifier"])
    parser.add_argument("--db-path", type=str, required=True)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--plot-path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.study_name is not None:
        study_name = args.study_name
    elif args.model_type == "unet":
        study_name = "unet_full_tuning"
    else:
        study_name = "classifier_full_tuning"

    study_matrix, params, loss_curves, best_id = load_study_data(args.db_path, study_name)
    create_csv_report(study_matrix, params, args.csv_path)
    plot_from_results(args.csv_path, loss_curves, best_id, args.plot_path, args.model_type)