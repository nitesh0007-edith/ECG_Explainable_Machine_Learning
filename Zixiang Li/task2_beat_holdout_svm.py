import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

print("loading...")
data = np.load('task1_processed_data.npz')

X_train = data['X_train_beats']
y_train = data['y_train_beats']
X_test = data['X_test_beats']
y_test = data['y_test_beats']

print(f"training data: {X_train.shape}")
print(f"testing data: {X_test.shape}")

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

def evaluate_model(y_true, y_pred, model_name="Model"):
    print(f"\n{'=' * 60}")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    categories = ['N', 'L', 'R', 'V', 'A', 'F', 'f', '/']
    print(classification_report(y_true, y_pred, target_names=categories, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories,
                yticklabels=categories,
                cbar_kws={'label': 'Count'})
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    filename = f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

results = evaluate_model(y_test, y_pred, "SVM_Beat_Holdout")

np.savez('task2_svm_beat_results.npz',
         y_test=y_test,
         y_pred=y_pred,
         accuracy=results['accuracy'],
         precision=results['precision'],
         recall=results['recall'],
         f1_score=results['f1_score'],
         confusion_matrix=results['confusion_matrix'])


