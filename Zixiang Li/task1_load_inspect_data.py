import numpy as np
from sklearn.preprocessing import OneHotEncoder


train_beats = np.loadtxt('train_beats.csv', delimiter=',')
test_beats = np.loadtxt('test_beats.csv', delimiter=',')
train_patients = np.loadtxt('train_patients.csv', delimiter=',')
test_patients = np.loadtxt('test_patients.csv', delimiter=',')

# Beat Holdout
X_train_beats = train_beats[:, :-2]
y_train_beats = train_beats[:, -2]
X_test_beats = test_beats[:, :-2]
y_test_beats = test_beats[:, -2]

# Patient Holdout
X_train_patients = train_patients[:, :-2]
y_train_patients = train_patients[:, -2]
X_test_patients = test_patients[:, :-2]
y_test_patients = test_patients[:, -2]

encoder = OneHotEncoder(sparse_output=False)
y_train_beats_onehot = encoder.fit_transform(y_train_beats.reshape(-1, 1))
y_test_beats_onehot = encoder.transform(y_test_beats.reshape(-1, 1))

encoder_patients = OneHotEncoder(sparse_output=False)
y_train_patients_onehot = encoder_patients.fit_transform(y_train_patients.reshape(-1, 1))
y_test_patients_onehot = encoder_patients.transform(y_test_patients.reshape(-1, 1))

print("Beat Holdout:")
print(f"  Train: X={X_train_beats.shape}, y={y_train_beats.shape}")
print(f"  Test:  X={X_test_beats.shape}, y={y_test_beats.shape}")

print("\nPatient Holdout:")
print(f"  Train: X={X_train_patients.shape}, y={y_train_patients.shape}")
print(f"  Test:  X={X_test_patients.shape}, y={y_test_patients.shape}")
print(f"  Beat train: {y_train_beats_onehot.shape}")
print(f"  Patient train: {y_train_patients_onehot.shape}")

np.savez('task1_processed_data.npz',
         X_train_beats=X_train_beats,
         y_train_beats=y_train_beats,
         y_train_beats_onehot=y_train_beats_onehot,
         X_test_beats=X_test_beats,
         y_test_beats=y_test_beats,
         y_test_beats_onehot=y_test_beats_onehot,
         X_train_patients=X_train_patients,
         y_train_patients=y_train_patients,
         y_train_patients_onehot=y_train_patients_onehot,
         X_test_patients=X_test_patients,
         y_test_patients=y_test_patients,
         y_test_patients_onehot=y_test_patients_onehot)

