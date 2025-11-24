import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def permutation_feature_importance(X, y, n_folds=5, n_slices=11, random_state=42):

    n_samples, n_features = X.shape
    slice_size = n_features // n_slices
    importance_matrix = np.zeros((n_folds, n_slices))
    # K-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold_idx + 1}/{n_folds}...", end=" ")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=random_state)
        svm.fit(X_train, y_train)

        y_pred_original = svm.predict(X_test)
        original_acc = accuracy_score(y_test, y_pred_original)

        for slice_idx in range(n_slices):
            start_col = slice_idx * slice_size
            end_col = start_col + slice_size
            X_test_permuted = X_test.copy()
            np.random.seed(random_state + fold_idx * n_slices + slice_idx)
            for col in range(start_col, end_col):
                np.random.shuffle(X_test_permuted[:, col])

            y_pred_permuted = svm.predict(X_test_permuted)
            permuted_acc = accuracy_score(y_test, y_pred_permuted)
            importance_score = original_acc - permuted_acc
            importance_matrix[fold_idx, slice_idx] = importance_score

    mean_importance = np.mean(importance_matrix, axis=0)

    return mean_importance

def plot_importance(importance, title, filename):
    n_slices = len(importance)
    slice_size = 275 // n_slices
    slice_labels = [f"{i * slice_size}-{(i + 1) * slice_size}" for i in range(n_slices)]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(n_slices), importance, color='steelblue', alpha=0.8, edgecolor='black')
    max_idx = np.argmax(importance)
    bars[max_idx].set_color('crimson')

    plt.xlabel('Slice Index (Time Points)', fontsize=12, fontweight='bold')
    plt.ylabel('Importance Score\n(Accuracy Drop)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(n_slices), slice_labels, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    plt.text(max_idx, importance[max_idx],
             f'  Most Important\n  Score: {importance[max_idx]:.4f}',
             ha='left', va='bottom', fontsize=10, color='crimson', fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

data = np.load('task1_processed_data.npz')
X_train_beats = data['X_train_beats']
y_train_beats = data['y_train_beats']

importance_beats = permutation_feature_importance(
    X_train_beats, y_train_beats,
    n_folds=5, n_slices=11, random_state=42
)

for i, score in enumerate(importance_beats):
    print(f"  切片 {i}: {score:.4f}")
plot_importance(
    importance_beats,
    'Permutation Feature Importance - Beat Holdout SVM',
    'task4_importance_beat_holdout.png'
)

X_train_patients = data['X_train_patients']
y_train_patients = data['y_train_patients']

sample_size = 30000
np.random.seed(42)
indices = np.random.choice(len(X_train_patients), sample_size, replace=False)
X_sample = X_train_patients[indices]
y_sample = y_train_patients[indices]

importance_patients = permutation_feature_importance(
    X_sample, y_sample,
    n_folds=5, n_slices=11, random_state=42
)

for i, score in enumerate(importance_patients):
    print(f"  slice {i}: {score:.4f}")

plot_importance(
    importance_patients,
    'Permutation Feature Importance - Patient Holdout SVM',
    'task4_importance_patient_holdout.png'
)

np.savez('task4_importance_results.npz',
         importance_beats=importance_beats,
         importance_patients=importance_patients)

