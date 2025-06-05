# 🧠 Heart Disease Prediction using Decision Trees and Random Forests

This project demonstrates how to use **Decision Tree** and **Random Forest** classifiers to predict the presence of heart disease in patients. We use the `heart.csv` dataset and visualize the decision tree model using Graphviz.

---

## 📌 Project Objective

- Train a **Decision Tree Classifier** and visualize the tree.
- Analyze **overfitting** and tune hyperparameters like tree depth.
- Train a **Random Forest Classifier** and compare performance.
- Evaluate using **confusion matrix**, **classification report**, and **cross-validation**.
- Understand **feature importance** in predictions.

---

## 📁 Project Structure

DecisionTreeProject/
│
├── heart.csv # Dataset file
├── README.md # Project description
├── .gitignore # Git ignore file
├── main.py # Main execution script
│
├── src/ # Source code files
│ ├── data_loader.py # Data loading and preprocessing
│ ├── decision_tree_model.py # Training + visualizing Decision Tree
│ ├── random_forest_model.py # Training Random Forest
│ ├── evaluate.py # Evaluation functions
│ └── visualize.py # Visualization helpers (Graphviz)

yaml
Copy
Edit

---

## 🔧 Requirements

- Python 3.8+
- `pandas`, `scikit-learn`, `matplotlib`, `graphviz`

Install all dependencies using:

```bash
pip install -r requirements.txt
To use Graphviz for visualization, make sure you install Graphviz binaries:

bash
Copy
Edit
# Windows
choco install graphviz
# or manually download from https://graphviz.org/download/

# Linux/macOS
sudo apt install graphviz
🚀 Running the Project
bash
Copy
Edit
python main.py
Sample output:

lua
Copy
Edit
Decision Tree Accuracy: 83.4%
Random Forest Accuracy: 85.7%
Confusion Matrix:
[[121  38]
 [ 13 136]]
Classification Report:
...
A decision_tree.pdf will be generated showing the trained decision tree.

📊 What You'll Learn
Decision Trees and how they split data based on features

How Random Forests reduce overfitting

Importance of cross-validation and evaluation metrics

Visualizing ML models to interpret decisions

Feature importance ranking

📈 Sample Decision Tree Output

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

📄 License
MIT License. Do whatever you want with it 😊

💡 Acknowledgements
Scikit-learn

UCI Heart Disease Dataset

Graphviz
