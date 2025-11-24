import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

data = np.load('task1_processed_data.npz')

X_train = data['X_train_patients']
y_train = data['y_train_patients']
X_test = data['X_test_patients']
y_test = data['y_test_patients']

unique_test = np.unique(y_test)
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

    unique_labels = np.unique(y_true)
    categories_all = ['N', 'L', 'R', 'V', 'A', 'F', 'f', '/']
    categories_present = [categories_all[int(label) - 1] for label in unique_labels]
    print(classification_report(y_true, y_pred,
                                target_names=categories_present,
                                labels=unique_labels,
                                zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories_present,
                yticklabels=categories_present,
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

results = evaluate_model(y_test, y_pred, "SVM_Patient_Holdout")
np.savez('task3_svm_patient_results.npz',
         y_test=y_test,
         y_pred=y_pred,
         accuracy=results['accuracy'],
         precision=results['precision'],
         recall=results['recall'],
         f1_score=results['f1_score'],
         confusion_matrix=results['confusion_matrix'])
