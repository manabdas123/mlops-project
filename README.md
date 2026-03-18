# 🚀 End-to-End MLOps Pipeline with Docker

## 📌 Project Overview

This project demonstrates a complete **MLOps pipeline** using a machine learning model deployed through a Flask API and containerized using Docker.

The goal of this project is to show how a machine learning model can be built, tracked, deployed, and served in a production-like environment.

---

## ⚙️ Tech Stack

* Python
* Scikit-learn
* Flask
* MLflow
* Docker
* Git & GitHub

---

## 📊 Workflow

1. Data Collection
2. Data Preprocessing
3. Model Training (Random Forest)
4. Experiment Tracking using MLflow
5. Model Saving (`model.pkl`)
6. API Creation using Flask
7. Containerization using Docker
8. Deployment & Testing

---

## 📁 Project Structure

```
mlops-project/
│
├── data/                # Dataset
├── src/                 # Training script
├── mlruns/              # MLflow tracking
├── app.py               # Flask API
├── test_api.py          # API testing script
├── model.pkl            # Trained model
├── requirements.txt     # Dependencies
├── Dockerfile           # Docker configuration
└── README.md            # Project documentation
```

---

## 🚀 How to Run the Project

### 🔹 1. Clone the repository

```
git clone https://github.com/your-username/mlops-project.git
cd mlops-project
```

---

### 🔹 2. Install dependencies

```
pip install -r requirements.txt
```

---

### 🔹 3. Run the Flask API

```
python app.py
```

Open:

```
http://127.0.0.1:5001
```

---

### 🔹 4. Run using Docker

#### Build image:

```
docker build -t mlops-app .
```

#### Run container:

```
docker run -p 5001:5001 mlops-app
```

---

## 🔮 API Usage

### Endpoint:

```
POST /predict
```

### Example Request:

```json
{
  "features": [3, 0, 22, 7.25]
}
```

### Example Response:

```json
{
  "prediction": 0
}
```

---

## 📈 Future Improvements

* Add CI/CD pipeline
* Deploy on cloud (AWS / GCP)
* Add frontend UI
* Model monitoring

---

## 👨‍💻 Author

**Manab Das**
