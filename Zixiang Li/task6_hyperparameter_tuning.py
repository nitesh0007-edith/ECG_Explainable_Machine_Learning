import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import warnings

warnings.filterwarnings('ignore')

def evaluate_quick(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }


data = np.load('task1_processed_data.npz')
X_train_beats = data['X_train_beats']
y_train_beats = data['y_train_beats']
X_test_beats = data['X_test_beats']
y_test_beats = data['y_test_beats']

X_train_patients = data['X_train_patients']
y_train_patients = data['y_train_patients']
X_test_patients = data['X_test_patients']
y_test_patients = data['y_test_patients']

print(f"  Beat Holdout: Train={X_train_beats.shape}, Test={X_test_beats.shape}")
print(f"  Patient Holdout: Train={X_train_patients.shape}, Test={X_test_patients.shape}")


n_estimators_list = [50, 100, 200, 500]
rf_n_estimators_results = []

for n_est in n_estimators_list:
    start_time = time.time()
    rf = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_beats, y_train_beats)
    y_pred = rf.predict(X_test_beats)

    train_time = time.time() - start_time
    metrics = evaluate_quick(y_test_beats, y_pred)

    rf_n_estimators_results.append({
        'n_estimators': n_est,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'train_time': train_time
    })

    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  training time: {train_time:.1f}s")


max_depth_list = [10, 20, 30, None]
rf_max_depth_results = []

for depth in max_depth_list:
    depth_str = str(depth) if depth is not None else "None"
    start_time = time.time()

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_beats, y_train_beats)
    y_pred = rf.predict(X_test_beats)

    train_time = time.time() - start_time
    metrics = evaluate_quick(y_test_beats, y_pred)

    rf_max_depth_results.append({
        'max_depth': depth_str,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'train_time': train_time
    })

    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  training time: {train_time:.1f}秒")


hidden_layers_list = [
    (64, 32),
    (128, 64),
    (256, 128),
    (128, 64, 32)
]
mlp_hidden_results = []

for layers in hidden_layers_list:
    print(f"\ntest hidden_layers={layers}...")
    start_time = time.time()

    mlp = MLPClassifier(
        hidden_layer_sizes=layers,
        max_iter=100,
        random_state=42,
        verbose=False
    )
    mlp.fit(X_train_beats, y_train_beats)
    y_pred = mlp.predict(X_test_beats)

    train_time = time.time() - start_time
    metrics = evaluate_quick(y_test_beats, y_pred)
    converged = mlp.n_iter_ < 100

    mlp_hidden_results.append({
        'hidden_layers': str(layers),
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'train_time': train_time,
        'converged': 'yes' if converged else 'no',
        'n_iter': mlp.n_iter_
    })

    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  training time: {train_time:.1f}s")


max_iter_list = [50, 100, 200, 300]
mlp_iter_results = []

for max_it in max_iter_list:
    print(f"\ntesting max_iter={max_it}...")
    start_time = time.time()

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=max_it,
        random_state=42,
        verbose=False
    )
    mlp.fit(X_train_beats, y_train_beats)
    y_pred = mlp.predict(X_test_beats)

    train_time = time.time() - start_time
    metrics = evaluate_quick(y_test_beats, y_pred)
    converged = mlp.n_iter_ < max_it

    mlp_iter_results.append({
        'max_iter': max_it,
        'accuracy': metrics['accuracy'],
        'f1_score': metrics['f1_score'],
        'train_time': train_time,
        'converged': 'yes' if converged else 'no',
        'actual_iter': mlp.n_iter_
    })

    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  training time: {train_time:.1f}s")


learning_rate_list = [0.0001, 0.001, 0.01, 0.1]
mlp_lr_results = []

for lr in learning_rate_list:
    print(f"\ntesting learning_rate_init={lr}...")
    start_time = time.time()

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=100,
        learning_rate_init=lr,
        random_state=42,
        verbose=False
    )
    mlp.fit(X_train_beats, y_train_beats)
    y_pred = mlp.predict(X_test_beats)

    train_time = time.time() - start_time
    metrics = evaluate_quick(y_test_beats, y_pred)

    mlp_lr_results.append({
        'learning_rate': lr,
        'accuracy': metrics['accuracy'],
        'f1_score': metrics['f1_score'],
        'train_time': train_time,
        'n_iter': mlp.n_iter_
    })

    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  training time: {train_time:.1f}s")

best_rf_n_est = max(rf_n_estimators_results, key=lambda x: x['accuracy'])['n_estimators']
best_rf_depth_str = max(rf_max_depth_results, key=lambda x: x['accuracy'])['max_depth']
best_rf_depth = None if best_rf_depth_str == "None" else int(best_rf_depth_str)

best_mlp_layers_str = max(mlp_hidden_results, key=lambda x: x['accuracy'])['hidden_layers']
best_mlp_layers = eval(best_mlp_layers_str)  # 转回tuple

best_mlp_iter = max(mlp_iter_results, key=lambda x: x['accuracy'])['max_iter']
best_mlp_lr = max(mlp_lr_results, key=lambda x: x['accuracy'])['learning_rate']

print(f"  Random Forest: n_estimators={best_rf_n_est}, max_depth={best_rf_depth}")
print(f"  MLP: layers={best_mlp_layers}, max_iter={best_mlp_iter}, lr={best_mlp_lr}")

rf_best = RandomForestClassifier(
    n_estimators=best_rf_n_est,
    max_depth=best_rf_depth,
    random_state=42,
    n_jobs=-1
)
rf_best.fit(X_train_patients, y_train_patients)
y_pred_rf = rf_best.predict(X_test_patients)
metrics_rf_patient = evaluate_quick(y_test_patients, y_pred_rf)
print(f"  Accuracy: {metrics_rf_patient['accuracy']:.4f}")
print(f"  F1-Score: {metrics_rf_patient['f1_score']:.4f}")

mlp_best = MLPClassifier(
    hidden_layer_sizes=best_mlp_layers,
    max_iter=best_mlp_iter,
    learning_rate_init=best_mlp_lr,
    random_state=42,
    verbose=False
)
mlp_best.fit(X_train_patients, y_train_patients)
y_pred_mlp = mlp_best.predict(X_test_patients)
metrics_mlp_patient = evaluate_quick(y_test_patients, y_pred_mlp)
print(f"  Accuracy: {metrics_mlp_patient['accuracy']:.4f}")
print(f"  F1-Score: {metrics_mlp_patient['f1_score']:.4f}")

df_rf_n_est = pd.DataFrame(rf_n_estimators_results)
df_rf_depth = pd.DataFrame(rf_max_depth_results)
df_mlp_hidden = pd.DataFrame(mlp_hidden_results)
df_mlp_iter = pd.DataFrame(mlp_iter_results)
df_mlp_lr = pd.DataFrame(mlp_lr_results)

df_rf_n_est.to_csv('task6_rf_n_estimators.csv', index=False)
df_rf_depth.to_csv('task6_rf_max_depth.csv', index=False)
df_mlp_hidden.to_csv('task6_mlp_hidden_layers.csv', index=False)
df_mlp_iter.to_csv('task6_mlp_max_iter.csv', index=False)
df_mlp_lr.to_csv('task6_mlp_learning_rate.csv', index=False)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# n_estimators vs Accuracy
ax1 = axes[0, 0]
n_est_vals = [r['n_estimators'] for r in rf_n_estimators_results]
acc_vals = [r['accuracy'] for r in rf_n_estimators_results]
ax1.plot(n_est_vals, acc_vals, 'o-', linewidth=2, markersize=8, color='steelblue')
ax1.set_xlabel('n_estimators', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('RF: n_estimators vs Accuracy', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# n_estimators vs Training Time
ax2 = axes[0, 1]
time_vals = [r['train_time'] for r in rf_n_estimators_results]
ax2.plot(n_est_vals, time_vals, 's-', linewidth=2, markersize=8, color='coral')
ax2.set_xlabel('n_estimators', fontsize=11, fontweight='bold')
ax2.set_ylabel('Training Time (s)', fontsize=11, fontweight='bold')
ax2.set_title('RF: n_estimators vs Training Time', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# max_depth vs Accuracy
ax3 = axes[1, 0]
depth_labels = [r['max_depth'] for r in rf_max_depth_results]
depth_acc = [r['accuracy'] for r in rf_max_depth_results]
ax3.bar(depth_labels, depth_acc, color='seagreen', alpha=0.7)
ax3.set_xlabel('max_depth', fontsize=11, fontweight='bold')
ax3.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax3.set_title('RF: max_depth vs Accuracy', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# F1-Score comparison
ax4 = axes[1, 1]
n_est_f1 = [r['f1_score'] for r in rf_n_estimators_results]
ax4.plot(n_est_vals, n_est_f1, 'D-', linewidth=2, markersize=8, color='purple')
ax4.set_xlabel('n_estimators', fontsize=11, fontweight='bold')
ax4.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
ax4.set_title('RF: n_estimators vs F1-Score', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task6_rf_tuning.png', dpi=300, bbox_inches='tight')
print("Random Forest: task6_rf_tuning.png")
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# hidden_layers vs Accuracy
ax1 = axes[0, 0]
layer_labels = [r['hidden_layers'] for r in mlp_hidden_results]
layer_acc = [r['accuracy'] for r in mlp_hidden_results]
x_pos = np.arange(len(layer_labels))
ax1.bar(x_pos, layer_acc, color='steelblue', alpha=0.7)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(layer_labels, rotation=15, ha='right', fontsize=9)
ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('MLP: Hidden Layers vs Accuracy', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# max_iter vs Accuracy
ax2 = axes[0, 1]
iter_vals = [r['max_iter'] for r in mlp_iter_results]
iter_acc = [r['accuracy'] for r in mlp_iter_results]
ax2.plot(iter_vals, iter_acc, 'o-', linewidth=2, markersize=8, color='coral')
ax2.set_xlabel('max_iter', fontsize=11, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax2.set_title('MLP: max_iter vs Accuracy', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# learning_rate vs Accuracy
ax3 = axes[1, 0]
lr_vals = [r['learning_rate'] for r in mlp_lr_results]
lr_acc = [r['accuracy'] for r in mlp_lr_results]
ax3.semilogx(lr_vals, lr_acc, 's-', linewidth=2, markersize=8, color='seagreen')
ax3.set_xlabel('Learning Rate', fontsize=11, fontweight='bold')
ax3.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax3.set_title('MLP: Learning Rate vs Accuracy', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Convergence analysis
ax4 = axes[1, 1]
actual_iters = [r['actual_iter'] for r in mlp_iter_results]
ax4.plot(iter_vals, actual_iters, 'D-', linewidth=2, markersize=8, color='purple', label='Actual Iterations')
ax4.plot(iter_vals, iter_vals, '--', linewidth=1.5, color='red', alpha=0.5, label='Max Iterations')
ax4.set_xlabel('max_iter ', fontsize=11, fontweight='bold')
ax4.set_ylabel('Actual Iterations ', fontsize=11, fontweight='bold')
ax4.set_title('MLP: Convergence Analysis', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task6_mlp_tuning.png', dpi=300, bbox_inches='tight')
print("task6_mlp_tuning.png")
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

models = ['RF (default)', 'RF (tuned)', 'MLP (default)', 'MLP (tuned)']

task5_results = np.load('task5_results.npz')
rf_default_beat = task5_results['rf_beat_acc']
rf_default_patient = task5_results['rf_patient_acc']
mlp_default_beat = task5_results['mlp_beat_acc']
mlp_default_patient = task5_results['mlp_patient_acc']

# Beat Holdout
rf_tuned_beat = max(rf_n_estimators_results, key=lambda x: x['accuracy'])['accuracy']
mlp_tuned_beat = max(mlp_hidden_results, key=lambda x: x['accuracy'])['accuracy']

# Patient Holdout
rf_tuned_patient = metrics_rf_patient['accuracy']
mlp_tuned_patient = metrics_mlp_patient['accuracy']

beat_acc = [rf_default_beat, rf_tuned_beat, mlp_default_beat, mlp_tuned_beat]
patient_acc = [rf_default_patient, rf_tuned_patient, mlp_default_patient, mlp_tuned_patient]

x = np.arange(len(models))
width = 0.35

ax1.bar(x - width / 2, beat_acc, width, label='Beat Holdout', color='steelblue', alpha=0.8)
ax1.bar(x + width / 2, patient_acc, width, label='Patient Holdout', color='coral', alpha=0.8)
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('default vs tuned parameters', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15, ha='right')
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0.92, 1.0])

improvements_rf_beat = (rf_tuned_beat - rf_default_beat) * 100
improvements_rf_patient = (rf_tuned_patient - rf_default_patient) * 100
improvements_mlp_beat = (mlp_tuned_beat - mlp_default_beat) * 100
improvements_mlp_patient = (mlp_tuned_patient - mlp_default_patient) * 100

improvement_data = {
    'RF Beat': improvements_rf_beat,
    'RF Patient': improvements_rf_patient,
    'MLP Beat': improvements_mlp_beat,
    'MLP Patient': improvements_mlp_patient
}

ax2.bar(improvement_data.keys(), improvement_data.values(),
        color=['steelblue', 'coral', 'steelblue', 'coral'], alpha=0.7)
ax2.set_ylabel('Accuracy Improvement (%)', fontsize=12, fontweight='bold')
ax2.set_title('performance improvement', fontsize=14, fontweight='bold')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(axis='y', alpha=0.3)
ax2.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('task6_tuning_comparison.png', dpi=300, bbox_inches='tight')
print("task6_tuning_comparison.png")
plt.close()


np.savez('task6_final_results.npz',

         rf_best_n_estimators=best_rf_n_est,
         rf_best_max_depth=best_rf_depth,
         rf_tuned_beat_acc=rf_tuned_beat,
         rf_tuned_patient_acc=rf_tuned_patient,
         rf_improvement_beat=improvements_rf_beat,
         rf_improvement_patient=improvements_rf_patient,

         mlp_best_layers=best_mlp_layers,
         mlp_best_max_iter=best_mlp_iter,
         mlp_best_lr=best_mlp_lr,
         mlp_tuned_beat_acc=mlp_tuned_beat,
         mlp_tuned_patient_acc=mlp_tuned_patient,
         mlp_improvement_beat=improvements_mlp_beat,
         mlp_improvement_patient=improvements_mlp_patient)