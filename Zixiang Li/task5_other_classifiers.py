import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

def evaluate_model(y_true, y_pred, model_name="Model", save_cm=True):
    print(f"\n{'=' * 60}")
    print(f"{model_name}")
    print(f"{'=' * 60}")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    if save_cm:
        unique_labels = np.unique(y_true)
        categories_all = ['N', 'L', 'R', 'V', 'A', 'F', 'f', '/']
        categories_present = [categories_all[int(label) - 1] for label in unique_labels]

        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=categories_present,
                    yticklabels=categories_present,
                    cbar_kws={'label': 'Count'})
        plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        filename = f'{model_name.lower().replace(" ", "_").replace("-", "_")}_cm.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def permutation_importance_fast(model, X, y, n_slices=11, n_repeats=3):
    n_features = X.shape[1]
    slice_size = n_features // n_slices
    y_pred_original = model.predict(X)
    original_acc = accuracy_score(y, y_pred_original)

    importance_scores = []

    for slice_idx in range(n_slices):
        start_col = slice_idx * slice_size
        end_col = start_col + slice_size

        slice_scores = []
        for repeat in range(n_repeats):
            X_permuted = X.copy()
            np.random.seed(42 + repeat)
            for col in range(start_col, end_col):
                np.random.shuffle(X_permuted[:, col])

            y_pred_permuted = model.predict(X_permuted)
            permuted_acc = accuracy_score(y, y_pred_permuted)
            slice_scores.append(original_acc - permuted_acc)

        importance_scores.append(np.mean(slice_scores))

    return np.array(importance_scores)


# ===== 可视化重要性 =====
def plot_importance_comparison(importance_dict, title, filename):
    n_slices = 11
    slice_labels = [f"{i * 25}-{(i + 1) * 25}" for i in range(n_slices)]

    plt.figure(figsize=(14, 6))
    x = np.arange(n_slices)
    width = 0.25

    colors = ['steelblue', 'coral', 'seagreen']
    for idx, (name, importance) in enumerate(importance_dict.items()):
        offset = (idx - 1) * width
        plt.bar(x + offset, importance, width, label=name, alpha=0.8, color=colors[idx])

    plt.xlabel('Slice Index (Time Points)', fontsize=12, fontweight='bold')
    plt.ylabel('Importance Score', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(x, slice_labels, rotation=45, ha='right')
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

data = np.load('task1_processed_data.npz')

X_train_beats = data['X_train_beats']
y_train_beats = data['y_train_beats']
X_test_beats = data['X_test_beats']
y_test_beats = data['y_test_beats']

X_train_patients = data['X_train_patients']
y_train_patients = data['y_train_patients']
X_test_patients = data['X_test_patients']
y_test_patients = data['y_test_patients']
print("Random Forest...")
rf_beat = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_beat.fit(X_train_beats, y_train_beats)

y_pred_rf_beat = rf_beat.predict(X_test_beats)
results_rf_beat = evaluate_model(y_test_beats, y_pred_rf_beat, "Random_Forest_Beat_Holdout")

print("Random Forest...")
rf_patient = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_patient.fit(X_train_patients, y_train_patients)

y_pred_rf_patient = rf_patient.predict(X_test_patients)
results_rf_patient = evaluate_model(y_test_patients, y_pred_rf_patient, "Random_Forest_Patient_Holdout")

print("MLP...")
mlp_beat = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50, random_state=42, verbose=False)
mlp_beat.fit(X_train_beats, y_train_beats)

y_pred_mlp_beat = mlp_beat.predict(X_test_beats)
results_mlp_beat = evaluate_model(y_test_beats, y_pred_mlp_beat, "MLP_Beat_Holdout")

print("MLP...")
mlp_patient = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50, random_state=42, verbose=False)
mlp_patient.fit(X_train_patients, y_train_patients)

y_pred_mlp_patient = mlp_patient.predict(X_test_patients)
results_mlp_patient = evaluate_model(y_test_patients, y_pred_mlp_patient, "MLP_Patient_Holdout")

sample_size = 5000
np.random.seed(42)
indices = np.random.choice(len(X_test_beats), sample_size, replace=False)
X_sample = X_test_beats[indices]
y_sample = y_test_beats[indices]

print("Random Forest's importance...")
importance_rf = permutation_importance_fast(rf_beat, X_sample, y_sample)

print("MLP's importance...")
importance_mlp = permutation_importance_fast(mlp_beat, X_sample, y_sample)

svm_importance = np.load('task4_importance_results.npz')['importance_beats']
importance_dict = {
    'SVM': svm_importance,
    'Random Forest': importance_rf,
    'MLP': importance_mlp
}

plot_importance_comparison(
    importance_dict,
    'Feature Importance Comparison - Beat Holdout',
    'task5_importance_comparison.png'
)

svm_beat_results = np.load('task2_svm_beat_results.npz')
svm_patient_results = np.load('task3_svm_patient_results.npz')
comparison_data = {
    'Model': ['SVM', 'Random Forest', 'MLP'],
    'Beat_Accuracy': [
        svm_beat_results['accuracy'],
        results_rf_beat['accuracy'],
        results_mlp_beat['accuracy']
    ],
    'Patient_Accuracy': [
        svm_patient_results['accuracy'],
        results_rf_patient['accuracy'],
        results_mlp_patient['accuracy']
    ]
}
for i in range(len(comparison_data['Model'])):
    model = comparison_data['Model'][i]
    beat_acc = comparison_data['Beat_Accuracy'][i]
    patient_acc = comparison_data['Patient_Accuracy'][i]
    diff = beat_acc - patient_acc
    print(f"{model:<20} {beat_acc:<15.4f} {patient_acc:<15.4f} {diff:<10.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
models = comparison_data['Model']
beat_acc = comparison_data['Beat_Accuracy']
patient_acc = comparison_data['Patient_Accuracy']

x = np.arange(len(models))
width = 0.35

ax1.bar(x - width / 2, beat_acc, width, label='Beat Holdout', color='steelblue')
ax1.bar(x + width / 2, patient_acc, width, label='Patient Holdout', color='coral')
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Model Comparison - Accuracy', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

diff = [b - p for b, p in zip(beat_acc, patient_acc)]
ax2.bar(models, diff, color='crimson', alpha=0.7)
ax2.set_ylabel('Accuracy Drop', fontsize=12, fontweight='bold')
ax2.set_title('Beat vs Patient - Performance Gap', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.savefig('task5_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

np.savez('task5_results.npz',
         rf_beat_acc=results_rf_beat['accuracy'],
         rf_patient_acc=results_rf_patient['accuracy'],
         mlp_beat_acc=results_mlp_beat['accuracy'],
         mlp_patient_acc=results_mlp_patient['accuracy'],
         importance_rf=importance_rf,
         importance_mlp=importance_mlp)
