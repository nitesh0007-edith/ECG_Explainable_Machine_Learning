#!/usr/bin/env python
# coding: utf-8

# In[2]:


import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.font_manager as fm
import numpy as np


# In[3]:


import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.font_manager as fm
import numpy as np

data_path = 'mit-bih-arrhythmia-database-1.0.0/'

all_record_names = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121', 
    '122', '123', '124', '200', '201', '202', '203', '204', '205', '206', 
    '207', '208', '209', '210', '212', '213', '214', '215', '217', '219',
    '220', '221', '222', '223', '228', '230', '231', '232', '233', '234'
]

class_mapping = {
    'N': 0,  # Normal
    'L': 1,  # Left Bundle Branch Block
    'R': 2,  # Right Bundle Branch Block  
    'V': 3,  # Premature Ventricular Contraction
    'A': 4,  # Atrial Premature Beat
    'F': 5,  # Fusion of Ventricular and Normal
    'f': 6,  # Fusion of Paced and Normal
    '/': 7   # Paced Beat
}

def extract_beats_from_record(record_name, samples_before=100, samples_after=174):
    """
    Extract beats from a single record
    """
    try:
        print(f"Processing record: {record_name}")
        record = wfdb.rdrecord(os.path.join(data_path, record_name))
        annotation = wfdb.rdann(os.path.join(data_path, record_name), 'atr')
        signal = record.p_signal[:, 0]
        beats = []
        labels = []

        for i in range(len(annotation.sample)):
            symbol = annotation.symbol[i]
            r_peak = annotation.sample[i]
            if symbol in class_mapping:
                start = r_peak - samples_before
                end = r_peak + samples_after
                if start >= 0 and end < len(signal):
                    beat = signal[start:end]
                    if np.std(beat) > 0:
                        beat = (beat - np.mean(beat)) / np.std(beat)
                    beats.append(beat)
                    labels.append(class_mapping[symbol])
        print(f"Record {record_name}: extracted {len(beats)} beats")
        return beats, labels, record_name 
    except Exception as e:
        print(f"Error processing record {record_name}: {e}")
        return [], [], record_name

print("Processing all 48 records from MIT-BIH database...")
all_beats = []
all_labels = []
all_patients = []

for i, record_name in enumerate(all_record_names):
    beats, labels, patient = extract_beats_from_record(record_name)
    all_beats.extend(beats)
    all_labels.extend(labels)
    all_patients.extend([patient] * len(beats))

    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1}/48 records")

if len(all_beats) > 0:
    X = np.array(all_beats)
    y = np.array(all_labels)
    patients = np.array(all_patients)
    
    print(f"\n✅ Successfully extracted {len(X)} beats")
    print(f"Data shape: {X.shape}")

    unique, counts = np.unique(y, return_counts=True)
    class_names = {v: k for k, v in class_mapping.items()}
    
    print("\nComplete class distribution:")
    total_beats = 0
    for class_id, count in zip(unique, counts):
        total_beats += count
        print(f"Class {class_names[class_id]} (ID: {class_id}): {count} samples ({count/total_beats*100:.2f}%)")
    
    print(f"\nTotal beats: {total_beats}")

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar([class_names[i] for i in unique], counts)
    plt.title('Heartbeat Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{count}', ha='center', va='bottom')
    
    plt.subplot(1, 2, 2)
    plt.pie(counts, labels=[class_names[i] for i in unique], autopct='%1.1f%%')
    plt.title('Class Proportion Distribution')
    
    plt.tight_layout()
    plt.savefig('complete_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(16, 10))
    for i, class_id in enumerate(unique):
        class_indices = np.where(y == class_id)[0]
        if len(class_indices) > 0:
            sample_idx = class_indices[0]
            plt.subplot(2, 4, i+1)
            plt.plot(X[sample_idx])
            plt.title(f'Class {class_names[class_id]} (ID: {class_id})\nSamples: {len(class_indices)}')
            plt.xlabel('Time Points')
            plt.ylabel('Normalized Amplitude')
            plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('complete_ecg_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

    test_patients = ['104', '113', '119', '208', '210'] 
    
    train_mask = ~np.isin(patients, test_patients)
    test_mask = np.isin(patients, test_patients)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    train_patients, test_patients = patients[train_mask], patients[test_mask]
    
    print(f"\nDataset split:")
    print(f"Training set size: {len(X_train)} (from {len(np.unique(train_patients))} patients)")
    print(f"Test set size: {len(X_test)} (from {len(np.unique(test_patients))} patients)")

    print(f"\nTraining set class distribution:")
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    for class_id, count in zip(train_unique, train_counts):
        print(f"Class {class_names[class_id]}: {count} samples")
    
    print(f"\nTest set class distribution:")
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    for class_id, count in zip(test_unique, test_counts):
        print(f"Class {class_names[class_id]}: {count} samples")

    def save_to_csv(X_data, y_data, patient_data, filename):
        data_with_patient = np.column_stack((X_data, y_data, [int(p) for p in patient_data]))
        df = pd.DataFrame(data_with_patient)

        column_names = [f'feature_{i}' for i in range(X_data.shape[1])] + ['label', 'patient_id']
        df.columns = column_names
        
        df.to_csv(filename, index=False)
        print(f"Saved: {filename} (contains {len(df)} rows, {len(df.columns)} columns)")

    save_to_csv(X_train, y_train, train_patients, 'mitbih_complete_train.csv')
    save_to_csv(X_test, y_test, test_patients, 'mitbih_complete_test.csv')

    stats_df = pd.DataFrame({
        'Class': [class_names[i] for i in unique],
        'Class_ID': unique,
        'Total_Count': counts,
        'Train_Count': [np.sum(y_train == i) for i in unique],
        'Test_Count': [np.sum(y_test == i) for i in unique]
    })
    stats_df.to_csv('dataset_statistics.csv', index=False)
    print("Saved: dataset_statistics.csv")

    print("\n🎉 Complete data processing finished!")
    print("Generated files:")
    print("- mitbih_complete_train.csv: Complete training data")
    print("- mitbih_complete_test.csv: Complete test data") 
    print("- complete_ecg_samples.png: All class samples visualization")
    print("- complete_class_distribution.png: Class distribution chart")
    print("- dataset_statistics.csv: Data statistics")
    
else:
    print("❌ No beats extracted, please check data path and files")


# In[4]:


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def load_complete_data(filename):
    """Load complete data"""
    df = pd.read_csv(filename)
    X = df.iloc[:, :-2].values
    y = df['label'].values
    patients = df['patient_id'].values
    return X, y, patients

print("Loading complete dataset...")
X_train, y_train, train_patients = load_complete_data('mitbih_complete_train.csv')
X_test, y_test, test_patients = load_complete_data('mitbih_complete_test.csv')

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

complete_class_names = {
    0: 'Normal (N)',
    1: 'LBBBB (L)', 
    2: 'RBBBB (R)',
    3: 'PVC (V)',
    4: 'APB (A)',
    5: 'Fusion (F)',
    6: 'Fusion Paced (f)',
    7: 'Paced (/)'
}

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data standardization completed!")

def train_and_evaluate_models():
    """Train and evaluate multiple models"""
    models = {
        'SVM-RBF': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        'SVM-Linear': SVC(kernel='linear', C=1.0, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        print(f"{name} accuracy: {accuracy:.4f}")
    
    return results

print("Starting training of multiple models...")
results = train_and_evaluate_models()

best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
best_accuracy = results[best_model_name]['accuracy']
print(f"\n🎯 Best model: {best_model_name}, accuracy: {best_accuracy:.4f}")

best_result = results[best_model_name]
y_pred_best = best_result['predictions']

print(f"\n{best_model_name} detailed evaluation:")
print("=" * 60)
print(classification_report(y_test, y_pred_best, 
                          target_names=[complete_class_names[i] for i in sorted(complete_class_names.keys())]))

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[complete_class_names[i] for i in sorted(complete_class_names.keys())],
            yticklabels=[complete_class_names[i] for i in sorted(complete_class_names.keys())])
plt.title(f'Confusion Matrix - {best_model_name}\nAccuracy: {best_accuracy:.4f}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('best_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ All model training completed!")


# In[5]:


def generate_statistics_report():
    """Generate detailed data statistics report"""
    stats_df = pd.read_csv('dataset_statistics.csv')
    
    print("=" * 50)
    print("MIT-BIH Dataset Complete Statistics Report")
    print("=" * 50)
    
    total_samples = stats_df['Total_Count'].sum()
    train_samples = stats_df['Train_Count'].sum()
    test_samples = stats_df['Test_Count'].sum()
    
    print(f"\nOverall Statistics:")
    print(f"Total beats: {total_samples}")
    print(f"Training set: {train_samples} ({train_samples/total_samples*100:.1f}%)")
    print(f"Test set: {test_samples} ({test_samples/total_samples*100:.1f}%)")
    print(f"Number of classes: {len(stats_df)}")
    
    print(f"\nDetailed class distribution:")
    for _, row in stats_df.iterrows():
        print(f"{row['Class']:15} (ID:{row['Class_ID']:1}): {row['Total_Count']:5} | "
              f"Train: {row['Train_Count']:5} | Test: {row['Test_Count']:5}")

    plt.figure(figsize=(12, 6))
    
    x = range(len(stats_df))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], stats_df['Train_Count'], width, label='Training Set', alpha=0.7)
    plt.bar([i + width/2 for i in x], stats_df['Test_Count'], width, label='Test Set', alpha=0.7)
    
    plt.xlabel('Class')
    plt.ylabel('Sample Count')
    plt.title('Training and Test Set Class Distribution')
    plt.xticks(x, stats_df['Class'], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('train_test_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

generate_statistics_report()


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def create_comprehensive_report():
    """Create comprehensive project report"""

    model_results = {
        'Random Forest': 0.9053,
        'SVM-RBF': 0.8784, 
        'SVM-Linear': 0.7642
    }

    dataset_stats = {
        'total_samples': 108829,
        'train_samples': 97271,
        'test_samples': 11558,
        'num_classes': 8,
        'train_patients': 43,
        'test_patients': 5
    }

    class_distribution = {
        'N': {'train': 67519, 'test': 7500, 'total': 75019},
        'L': {'train': 8072, 'test': 0, 'total': 8072},
        'R': {'train': 7255, 'test': 0, 'total': 7255},
        'V': {'train': 5497, 'test': 1632, 'total': 7129},
        'A': {'train': 2546, 'test': 0, 'total': 2546},
        'F': {'train': 420, 'test': 382, 'total': 802},
        'f': {'train': 316, 'test': 666, 'total': 982},
        '/': {'train': 5646, 'test': 1378, 'total': 7024}
    }
    
    print("=" * 70)
    print("MIT-BIH Arrhythmia Classification Project - Complete Summary Report")
    print("=" * 70)
    print("\n📋 Project Overview")
    print("-" * 40)
    print(f"• Problem Type: Multi-class ECG Arrhythmia Classification")
    print(f"• Data Scale: {dataset_stats['total_samples']:,} beats")
    print(f"• Training Set: {dataset_stats['train_samples']:,} samples (43 patients)")
    print(f"• Test Set: {dataset_stats['test_samples']:,} samples (5 patients)") 
    print(f"• Number of Classes: {dataset_stats['num_classes']} arrhythmia types")
    print("\n🏆 Model Performance Summary")
    print("-" * 40)
    best_model = max(model_results.items(), key=lambda x: x[1])
    for model, accuracy in sorted(model_results.items(), key=lambda x: x[1], reverse=True):
        star = " 🎯" if model == best_model[0] else ""
        print(f"• {model:<15}: {accuracy:.4f} ({accuracy*100:.2f}%){star}")
    print("\n📊 Data Characteristics Analysis")
    print("-" * 40)
    imbalance_ratio = class_distribution['N']['total'] / class_distribution['F']['total']
    print(f"• Data Imbalance Ratio: {imbalance_ratio:.1f}:1 (Most/Least)")
    print(f"• Missing Classes in Test Set: L, R, A (these anomalies only appear in training set patients)")
    print(f"• Rare Classes: F (802), f (982) - require special attention")
    print("\n💡 Key Findings and Recommendations")
    print("-" * 40)
    print(f"• Best Model: {best_model[0]} (Accuracy: {best_model[1]:.4f})")
    print("• Random Forest performs excellently, suitable for complex ECG data patterns")
    print("• Data split is reasonable, test set contains unseen patients ensuring generalization capability")
    print("• Next Focus: Optimize recognition performance for rare classes (F, f)")

def create_visualizations():
    """Create results visualizations"""
    models = ['Random Forest', 'SVM-RBF', 'SVM-Linear']
    accuracies = [0.9053, 0.8784, 0.7642]
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    colors = ['green', 'blue', 'red']
    bars = plt.bar(models, accuracies, color=colors, alpha=0.7)
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=15)
    plt.ylim(0.7, 0.95)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.subplot(2, 2, 2)
    classes = ['N', 'L', 'R', 'V', 'A', 'F', 'f', '/']
    train_counts = [67519, 8072, 7255, 5497, 2546, 420, 316, 5646]
    
    plt.bar(classes, train_counts, color='skyblue', alpha=0.7)
    plt.title('Training Set Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Sample Count')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 3)
    sizes = [97271, 11558]
    labels = [f'Training\n{97271:,}', f'Test\n{11558:,}']
    colors = ['lightblue', 'lightcoral']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Dataset Split Ratio', fontsize=14, fontweight='bold')

    plt.subplot(2, 2, 4)
    potential_improvement = 0.95 - 0.9053  # Assuming target 95%
    current_performance = 0.9053
    remaining = [current_performance, potential_improvement]
    labels = ['Current Performance', 'Improvement Potential']
    colors = ['lightgreen', 'orange']
    
    plt.pie(remaining, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Performance Improvement Analysis', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('project_summary_report.png', dpi=300, bbox_inches='tight')
    plt.show()

create_comprehensive_report()
create_visualizations()


# In[7]:


def create_simple_visualization():
    """Create simplified data visualization report"""

    train_df = pd.read_csv('mitbih_complete_train.csv')
    test_df = pd.read_csv('mitbih_complete_test.csv')

    total_samples = len(train_df) + len(test_df)
    train_samples = len(train_df)
    test_samples = len(test_df)
    
    print("📊 Data Statistics Report")
    print("=" * 50)
    print(f"Total samples: {total_samples:,}")
    print(f"Training set: {train_samples:,} ({train_samples/total_samples*100:.1f}%)")
    print(f"Test set: {test_samples:,} ({test_samples/total_samples*100:.1f}%)")
    print(f"Number of features: {train_df.shape[1]-2}")

    train_counts = train_df['label'].value_counts().sort_index()
    test_counts = test_df['label'].value_counts().sort_index()
    
    class_names = {
        0: 'Normal (N)',
        1: 'LBBBB (L)', 
        2: 'RBBBB (R)',
        3: 'PVC (V)',
        4: 'APB (A)',
        5: 'Fusion (F)',
        6: 'Fusion Paced (f)',
        7: 'Paced (/)'
    }
    
    print(f"\n🏥 Training Set Class Distribution:")
    for class_id in sorted(train_counts.index):
        count = train_counts[class_id]
        percentage = count / train_samples * 100
        print(f"  {class_names[class_id]}: {count:>6,} ({percentage:>5.1f}%)")
    
    print(f"\n🧪 Test Set Class Distribution:")
    for class_id in sorted(test_counts.index):
        count = test_counts[class_id]
        percentage = count / test_samples * 100
        print(f"  {class_names[class_id]}: {count:>6,} ({percentage:>5.1f}%)")

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.pie([train_samples, test_samples], 
            labels=[f'Train\n{train_samples:,}', f'Test\n{test_samples:,}'], 
            autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
    plt.title('Dataset Split')
    
    plt.subplot(1, 2, 2)
    if len(train_counts) > 0:
        display_classes = train_counts.head(6)
        plt.pie(display_classes.values, 
                labels=[class_names[i] for i in display_classes.index],
                autopct='%1.1f%%')
        plt.title('Training Set Class Distribution (Top 6)')
    
    plt.tight_layout()
    plt.savefig('simple_data_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Simplified report generated: simple_data_report.png")

create_simple_visualization()


# In[8]:


from sklearn.metrics import f1_score, precision_score, recall_score
import time

def comprehensive_evaluation(y_true, y_pred, model_name, training_time):
    
    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_precision = precision_score(y_true, y_pred, average='weighted')
    weighted_recall = recall_score(y_true, y_pred, average='weighted')
    
    results = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Weighted F1': weighted_f1,
        'Macro F1': macro_f1,
        'Weighted Precision': weighted_precision,
        'Weighted Recall': weighted_recall,
        'Training Time (s)': training_time
    }
    
    return results

def detailed_model_evaluation():
    
    models = {
        'SVM-RBF': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test_scaled)
        
        metrics = comprehensive_evaluation(y_test, y_pred, name, training_time)
        results.append(metrics)
        
        print(f"✅ {name} completed!")
        print(f"   Accuracy: {metrics['Accuracy']:.4f}")
        print(f"   Weighted F1: {metrics['Weighted F1']:.4f}")
        print(f"   Training time: {training_time:.2f} seconds")
    
    return pd.DataFrame(results)

print("Starting detailed model evaluation...")
detailed_results = detailed_model_evaluation()

print("\n📊 Detailed Model Performance Comparison")
print("="*50)
detailed_results_sorted = detailed_results.sort_values('Accuracy', ascending=False)
print(detailed_results_sorted.to_string(index=False))


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

def hyperparameter_tuning_two_models():
    """Hyperparameter tuning for two models"""
    
    print("\n" + "="*80)
    print("🎯 Hyperparameter Tuning - Two Models")
    print("="*80)
    
    models_to_tune = ['Random Forest', 'SVM-RBF']
    
    tuning_results = []
    
    for model_name in models_to_tune:
        print(f"\n🔧 Tuning {model_name}...")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=42)
            
        elif model_name == 'SVM-RBF':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
            base_model = SVC(kernel='rbf', random_state=42)

        print(f"   Performing random search...")
        start_time = time.time()
        random_search = RandomizedSearchCV(
            base_model, param_grid, n_iter=10, cv=3, 
            scoring='accuracy', n_jobs=-1, random_state=42
        )
        
        random_search.fit(X_train_scaled, y_train)
        tuning_time = time.time() - start_time
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        best_model = random_search.best_estimator_
        y_pred_tuned = best_model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred_tuned)
        weighted_f1 = f1_score(y_test, y_pred_tuned, average='weighted')
        tuning_results.append({
            'Model': f"{model_name} (Tuned)",
            'Best Parameters': str(best_params),
            'CV Score': best_score,
            'Test Accuracy': test_accuracy,
            'Weighted F1': weighted_f1,
            'Tuning Time (s)': tuning_time
        })
        
        print(f"   ✅ Tuning completed!")
        print(f"      Best parameters: {best_params}")
        print(f"      Cross-validation score: {best_score:.4f}")
        print(f"      Test set accuracy: {test_accuracy:.4f}")
        print(f"      Tuning time: {tuning_time:.2f} seconds")
    
    return pd.DataFrame(tuning_results)

tuning_results = hyperparameter_tuning_two_models()


# In[17]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import time

def efficient_permutation_feature_importance(model, X, y, n_slices=11, n_folds=3, 
                                           sample_size=3000, random_state=42):
    """
    Efficient permutation feature importance analysis - using data sampling and parallel processing
    """
    np.random.seed(random_state)

    if len(X) > sample_size:
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_indices]
        y_sample = y[sample_indices]
    else:
        X_sample = X
        y_sample = y
    
    slice_size = 25
    n_features = X_sample.shape[1]
    n_slices = min(n_slices, n_features // slice_size)
    
    print(f"Starting permutation feature importance analysis...")
    print(f"Data sample: {X_sample.shape}, Number of slices: {n_slices}, Number of folds: {n_folds}")
    
    feature_importance = np.zeros(n_slices)
    fold_scores = []

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_sample, y_sample)):
        print(f"  Processing fold {fold+1}/{n_folds}...")
        
        X_train, X_val = X_sample[train_idx], X_sample[val_idx]
        y_train, y_val = y_sample[train_idx], y_sample[val_idx]

        model_clone = clone(model)
        model_clone.fit(X_train, y_train)

        y_pred_original = model_clone.predict(X_val)
        baseline_score = accuracy_score(y_val, y_pred_original)
        
        fold_slice_scores = []

        for slice_idx in range(n_slices):
            start_idx = slice_idx * slice_size
            end_idx = min((slice_idx + 1) * slice_size, n_features)

            X_permuted = X_val.copy()

            for sample_idx in range(X_permuted.shape[0]):
                slice_data = X_permuted[sample_idx, start_idx:end_idx].copy()
                np.random.shuffle(slice_data)
                X_permuted[sample_idx, start_idx:end_idx] = slice_data

            y_pred_permuted = model_clone.predict(X_permuted)
            permuted_score = accuracy_score(y_val, y_pred_permuted)

            importance = baseline_score - permuted_score
            fold_slice_scores.append(importance)

            feature_importance[slice_idx] += importance
        
        fold_scores.append(fold_slice_scores)

    feature_importance /= n_folds
    fold_scores = np.array(fold_scores)
    feature_importance_std = np.std(fold_scores, axis=0)
    
    return feature_importance, feature_importance_std

def plot_feature_importance_comparison(rf_importance, svm_importance, std_rf, std_svm):
    """Compare permutation feature importance of two models"""
    
    n_slices = len(rf_importance)
    slices = [f'Slice {i+1}\n({i*25+1}-{(i+1)*25})' for i in range(n_slices)]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    bars1 = ax1.bar(range(n_slices), rf_importance, yerr=std_rf, 
                   capsize=5, alpha=0.7, color='skyblue', label='Random Forest')
    ax1.set_title('Random Forest - Permutation Feature Importance Analysis', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Feature Slice')
    ax1.set_ylabel('Importance Score')
    ax1.set_xticks(range(n_slices))
    ax1.set_xticklabels(slices, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    for i, (bar, importance) in enumerate(zip(bars1, rf_importance)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{importance:.4f}', ha='center', va='bottom', fontsize=9)

    bars2 = ax2.bar(range(n_slices), svm_importance, yerr=std_svm,
                   capsize=5, alpha=0.7, color='lightcoral', label='SVM-RBF')
    ax2.set_title('SVM-RBF - Permutation Feature Importance Analysis', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Feature Slice')
    ax2.set_ylabel('Importance Score')
    ax2.set_xticks(range(n_slices))
    ax2.set_xticklabels(slices, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    for i, (bar, importance) in enumerate(zip(bars2, svm_importance)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{importance:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('permutation_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n🔍 Feature Importance Key Findings:")
    print("-" * 50)

    rf_top_slice = np.argmax(rf_importance)
    svm_top_slice = np.argmax(svm_importance)
    
    print(f"Random Forest most important feature slice: Slice {rf_top_slice + 1}")
    print(f"SVM-RBF most important feature slice: Slice {svm_top_slice + 1}")

    correlation = np.corrcoef(rf_importance, svm_importance)[0, 1]
    print(f"Feature importance correlation between two models: {correlation:.4f}")
    
    if correlation > 0.7:
        print("✅ High agreement in feature importance between two models")
    elif correlation > 0.3:
        print("⚠️ Moderate agreement in feature importance between two models")
    else:
        print("❌ Low agreement in feature importance between two models")

print("Starting permutation feature importance analysis...")
start_time = time.time()

best_rf_model = results['Random Forest']['model']
best_svm_model = results['SVM-RBF']['model']

print("\n1. Analyzing Random Forest...")
rf_importance, rf_std = efficient_permutation_feature_importance(
    best_rf_model, X_train_scaled, y_train, n_slices=11, n_folds=3, sample_size=5000
)

print("\n2. Analyzing SVM-RBF...")
svm_importance, svm_std = efficient_permutation_feature_importance(
    best_svm_model, X_train_scaled, y_train, n_slices=11, n_folds=3, sample_size=5000
)

analysis_time = time.time() - start_time
print(f"\n✅ Permutation feature importance analysis completed! Total time: {analysis_time:.2f} seconds")

plot_feature_importance_comparison(rf_importance, svm_importance, rf_std, svm_std)


# In[18]:


def plot_ecg_arrythmia_examples(X, y, class_names, samples_per_class=2):
    """
    Plot ECG signal examples of different types of arrhythmias
    """
    existing_classes = []
    for class_idx in sorted(class_names.keys()):
        if np.sum(y == class_idx) > 0:
            existing_classes.append(class_idx)
    
    n_classes = len(existing_classes)
    
    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(15, 3*n_classes))
    
    if n_classes == 1:
        axes = np.array([axes])
    
    for row, class_idx in enumerate(existing_classes):
        class_mask = y == class_idx
        class_samples = X[class_mask]

        selected_indices = np.random.choice(
            len(class_samples), 
            min(samples_per_class, len(class_samples)), 
            replace=False
        )
        
        for col, sample_idx in enumerate(selected_indices):
            ax = axes[row, col] if n_classes > 1 else axes[col]
            ecg_signal = class_samples[sample_idx]

            time_points = np.arange(len(ecg_signal))
            ax.plot(time_points, ecg_signal, linewidth=1.5, color='blue', alpha=0.8)
            
            ax.set_title(f'{class_names[class_idx]} - Example {col+1}', fontweight='bold')
            ax.set_xlabel('Time Points')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)

            ax.set_ylim([-3, 3])
    
    plt.tight_layout()
    plt.savefig('ecg_arrythmia_examples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Plotted ECG examples for {n_classes} types of arrhythmias")

print("Plotting ECG signals of different types of arrhythmias...")
plot_ecg_arrythmia_examples(X_train, y_train, complete_class_names, samples_per_class=2)


# In[ ]:




