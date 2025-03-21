# Machine-Learning-Group-Project
Lingnan University - Machine Learning for Business - Group Project 
## Overview
This project aims to develop a diabetes detection system, contains two parts: front-end and back-end
- Front-end: use Vue Structure, responsible for user interface display and interaction.
- Back-end: use Python which contains a machine learning model(MLP) to make prediction.


## Installation
### Clone the repository:
```bash
git clone https://github.com/Hanzo-neverdie/Machine-Learning-Group-Project.git
cd Machine-Learning-Group-Project
```

### Let me introduce the files:
```text
project-root/
├── diabetes_detection/(VUE structrue)
│   ├── .vscode/
│   ├── node_modules/
│   ├── public/
│   ├── src/
│   ├── ...
├── backend/
│   ├── app.py(used to make predictation)
│   ├── best_pipeline.joblib(saved model by running ML_network.py)
│   └── ML_network.py
├── diabetes.csv(Pima Indians Diabetes from kaggle(https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download))
```
#### ML_network.py 
It demonstrates a complete process of machine learning using the Multilayer Perceptron (MLP) model:
- Accuracy: 0.7468
- Precision: 0.6531
- Recall: 0.5926
- Specificity: 0.8300
- F1 Score: 0.6214
- AUC: 0.8131  
![示例图片](ROC_Curve.png)
### Environment Setup
#### Prerequisites:
- python
- Code Editor(Recommended: Visual Studio Code)
- Node.js(Version 18.3 or higher)
#### Steps:
- Follow the Vue.js official [quick start](https://vuejs.org/guide/quick-start.html) to set up the front-end environment.
- Install additional libraries such as PyTorch, NumPy, etc.


## Deployment
### To start front-end code
```bash
cd diabetes_detection
npm run dev
```
### To start back-end code
```bash
cd ..
cd backend
python app.py
```
### Access the Website
Open your browser and navigate to http://localhost:5173/.

## Video Link
Video Demonstration - https://youtu.be/0ys5YT282GM



