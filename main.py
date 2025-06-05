from src.load_data import load_data, split_data
from src.preprocess import preprocess_data
from src.decision_tree_model import train_decision_tree, visualize_tree
from src.random_forest_model import train_random_forest
from src.evaluate import evaluate_model, cross_validate
from src.feature_importance import plot_feature_importances

def main():
    # Load and split
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Preprocess
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

    # Decision Tree
    dt_model = train_decision_tree(X_train_scaled, y_train, max_depth=4)
    acc, cm, report = evaluate_model(dt_model, X_test_scaled, y_test)
    print("Decision Tree Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

    # Visualize tree
    visualize_tree(dt_model, X_train.columns, ["No Disease", "Disease"])

    # Random Forest
    rf_model = train_random_forest(X_train_scaled, y_train, max_depth=4)
    acc_rf, cm_rf, report_rf = evaluate_model(rf_model, X_test_scaled, y_test)
    print("Random Forest Accuracy:", acc_rf)
    print("Confusion Matrix:\n", cm_rf)
    print("Classification Report:\n", report_rf)

    # Feature importances
    plot_feature_importances(rf_model, X_train.columns)

    # Cross-validation score
    scores = cross_validate(rf_model, X_train_scaled, y_train)
    print("Cross-validation scores (Random Forest):", scores)
    print("Average CV Accuracy:", scores.mean())

if __name__ == "__main__":
    main()
