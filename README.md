# 🌸 Iris Species Prediction App

Welcome to the **Iris Species Prediction App**, a simple Streamlit-based web app that predicts iris flower species based on user input or uploaded CSV data.

🔗 **Live Demo**: [https://iris-exploration-salsa.streamlit.app/](https://iris-exploration-salsa.streamlit.app/)

## 👩🏻‍💻 About the Creator

This app was created by **Salsabila Qonita Kaltsum** as part of a machine learning portfolio project. The purpose of this project is to explore how to build interactive applications using Streamlit and apply classification models using scikit-learn.

## 📝 Description

The app predicts the species of iris flowers using the classic **Iris dataset** from UCIML. Users can input data manually through the sidebar or upload a CSV file. The model will process the data and predict the flower species:

- *Iris-setosa*
- *Iris-versicolor*
- *Iris-virginica*

## 📁 Files Included

- `iris_app.py`: Main Streamlit app script.
- `generate_iris.pkl`: Pre-trained machine learning model saved using `pickle`.

## ⚙️ Requirements

Make sure the following libraries are installed:

```bash
pip install -r requirements.txt
```

**Contents of `requirements.txt`:**

```
streamlit
pandas
scikit-learn
```

## 🚀 Running Locally

1. Clone this repository.
2. Install all dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run iris_app.py
```

---

Thank you for visiting! 🌼
