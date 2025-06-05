from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

def train_decision_tree(X_train, y_train, max_depth=None):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def visualize_tree(model, feature_names, class_names):
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree")  # Creates a PDF
    return graph
