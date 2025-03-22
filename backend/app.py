from flask import Flask, request, jsonify
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from flask_cors import CORS  # import CORS
# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define custom preprocessor
class TorchPreprocessor(TransformerMixin, BaseEstimator):
    """Custom preprocessor component"""
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        # Ensure input is a NumPy array
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        # Ensure input is a NumPy array
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        X_scaled = self.scaler.transform(X)
        return torch.FloatTensor(X_scaled)

# define custom PyTorch estimator
class TorchEstimator(BaseEstimator):
    """PyTorch模型封装器"""
    def __init__(self, hidden_layers=[64,32], dropout=0.5, lr=0.001, epochs=100):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        
    def fit(self, X, y):
        # Ensure input is Tensor
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if isinstance(y, np.ndarray):
            y = torch.FloatTensor(y).reshape(-1, 1)
        
        # initialize model
        self.model = DiabetesMLP(
            input_size=X.shape[1],
            hidden_layers=self.hidden_layers,
            dropout=self.dropout
        ).to(device)
        
        # optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        
        # transform to DataLoader
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # train loop
        for _ in range(self.epochs):
            self.model.train()
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        return self
    
    def predict_proba(self, X):
        # Ensure input is Tensor
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X.to(device)).cpu().numpy()
        return np.concatenate([1-preds, preds], axis=1)

# define MLP model
class DiabetesMLP(nn.Module):
    def __init__(self, input_size=8, hidden_layers=[64, 32], dropout=0.5):
        super(DiabetesMLP, self).__init__()
        layers = []
        prev_size = input_size
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = size
        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return torch.sigmoid(self.net(x))

# load model
model_path = '.\\best_pipeline.joblib'
model = joblib.load(model_path)

app = Flask(__name__)
CORS(app)  # turn on CORS
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # get JSON data
        data = request.json
        input_array = np.array([
            data['Pregnancies'],
            data['Glucose'],
            data['BloodPressure'],
            data['SkinThickness'],
            data['Insulin'],
            data['BMI'],
            data['DiabetesPedigreeFunction'],
            data['Age']
        ]).reshape(1, -1)
        
        # make prediction
        prediction = model.predict_proba(input_array)
        result = 'Diabetic' if prediction[0][1] >= 0.5 else 'Non-Diabetic'
        probability = prediction[0][1]

        return jsonify({
            'result': result,
            'probability': float(probability)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)