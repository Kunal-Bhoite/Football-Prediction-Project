
# ⚽ Football Tournament Winner Prediction using Machine Learning

### 🔍 Master’s Research Project – Women’s Football Analytics

📍 Dublin Business School | MSc in Business Analytics
👤 Developed by: **Kunal Kiran Bhoite**

---

## 📌 Project Overview

This research project focuses on predicting the winners of international **women’s football tournaments**, including the latest **Women’s FIFA World Cup**, using machine learning.

Historical match data was collected and analyzed to identify key factors influencing match outcomes — such as team performance, rankings, venue, climate conditions, and tournament context.

Four machine learning models were trained and evaluated:

* **XGBoost**
* Logistic Regression
* Support Vector Machine (SVM)
* Gaussian Naïve Bayes

A **Flask-based web application** was built to allow users to:
✔ Predict the **World Cup Winner**
✔ Simulate full tournament results (Groups → Knockout → Final)
✔ Predict head-to-head match outcomes between any two teams

## 🎯 Objectives

* Develop accurate predictive models using women’s football match data
* Predict group winners, knockout qualifiers, and final match results
* Compare the performance of ML algorithms
* Deploy predictions with a user-friendly interactive web UI

---

## 🧠 Machine Learning Results

| Algorithm           | Accuracy   | Precision | Recall | F1-Score |
| ------------------- | ---------- | --------- | ------ | -------- |
| **XGBoost**         | **70.80%** | 0.696     | 0.757  | 0.696    |
| Logistic Regression | 68.9%      | 0.640     | 0.730  | 0.680    |
| Gaussian NB         | 65.1%      | 0.556     | 0.720  | 0.610    |
| SVM                 | 58.5%      | 0.630     | 0.560  | 0.590    |

🏆 **Predicted Tournament Winner: Switzerland**
*(Based on full Women’s World Cup simulation)*


## 🛠️ Tech Stack

| Domain          | Technologies             |
| --------------- | ------------------------ |
| Programming     | Python                   |
| ML Frameworks   | XGBoost, Scikit-Learn    |
| Data Processing | Pandas, NumPy            |
| Deployment      | Flask Web Application    |
| Visualization   | Matplotlib               |
| Tools           | Jupyter Notebook, Joblib |

---

## 📂 Dataset

Data was sourced from:

* Kaggle (historical women’s football match results)
* Custom tournament schedule data

✔ Preprocessed & feature-engineered
✔ Handling of categorical encoding & ranking data
✔ Weather and location-based context included

---

## 🌐 Web Application Features

* Predict match winner between **any two teams**
* Full **points table** generation for each group
* Automated knockout stage predictions
* Interactive user interface built using Flask & HTML templates

---

## 🚀 How to Run the Project

```sh
pip install -r requirements.txt
python app.py
```

Then open:
👉 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 📈 Importance & Applications

* Sports analytics & forecasting
* Broadcasting insights & predictions
* Betting & odds decision support
* Team performance strategy validation
* Fan engagement enhancement

---

## 🔮 Future Enhancements

🚧 Planned improvements:

* Cloud deployment for public access
* Real-time live match prediction
* Integration of:

  * Player performance metrics
  * Injury & squad rotation data
* Ensemble hybrid system for improved accuracy
* UI enhancement with dashboards & analytics metrics

---

## 📜 Academic Details

This project fulfills the Applied Research Project requirement for:
🎓 *Master of Science in Business Analytics*
🏫 **Dublin Business School**
📅 2024
👨‍🏫 Supervisor: *Mr. Paul Walsh*

---

## ⭐ Acknowledgments

Special thanks to:

* Dublin Business School for academic support
* My supervisor for continuous guidance
* Friends & family for encouragement

---

## 🏁 Conclusion

This research demonstrates the strong potential of **machine learning in sports analytics** — especially for predicting outcomes of major tournaments like the Women’s World Cup.
With **70.80% accuracy**, the XGBoost model proved to be the best performer, showing that **historical data can successfully forecast future match results**.




