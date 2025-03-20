# Machine-Learning-Group-Project
Lingnan University - Machine Learning for Business - Group Project 

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Hanzo-neverdie/Machine-Learning-Group-Project.git
cd Machine-Learning-Group-Project
```

Let me introduce the files:
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
```
### Environment Setup
Prerequisites:
- python
- Code Editor
- Node.js  
Then you can get a quick start with Vue.js official [quick start](https://vuejs.org/guide/quick-start.html).  
Also you need to install correlating libs, like pytorch, numpy and so on.

## Deployment
To start front-end code
```bash
cd diabetes_detection
npm run dev
```
To start back-end code
```bash
cd ..
cd backend
python app.py
```
Then you can view the website on http://localhost:5173/

