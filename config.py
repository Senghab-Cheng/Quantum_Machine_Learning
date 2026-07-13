from pathlib import Path
project_rooth = Path(__file__).resolve().parent.parent
Results_dir = project_rooth / "results"
classical_models_dir = Results_dir / "classical_models"
confusion_matrices_dir = Results_dir / "confusion_matrices"
Data_dir = project_rooth / "data"

Raw_data_path = Data_dir /"diabetes.csv"
for _dir in (Data_dir, Results_dir, classical_models_dir, confusion_matrices_dir):
    _dir.mkdir(parents=True, exist_ok=True)

    Target_column = "Outcome"
    feature_columns = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    Zero_As_missing_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    test_size  = 0.2
    random_state = 42

    CV_FOLDS = 5
    scoring = "f1"

    LOGISTIC_REGRESSION_PARAMS = { 
        "C": 1.0,
        "max_iter": 1000,
        "random_state": "RANDOM_STATE",
        "class_weight": "balanced",
        "solver": "lbfgs",
    }
    Decision_TREE_PARAMS = { 
        "max_depth": 5,
        "min_samples_split": 10,
        "random_state": random_state,
    }
    Logistic_regression_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["liblinear", "lbfgs"],
        "max_iter": [100, 500, 1000],
    }
    Decison_tree_grid = {
        "max_depth": [3, 5, 7,10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],

    }
    Random_forest_grid = {
        "n_esimators": [50, 100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2",],
        
    }


        
