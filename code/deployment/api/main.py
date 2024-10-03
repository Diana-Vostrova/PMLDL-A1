# File: code/deployment/api/main.py

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import io
from PIL import Image

app = FastAPI()

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out

# Model configuration
input_dim = 28    
hidden_dim = 100  
layer_dim = 1     
output_dim = 10   

# Load the pre-trained model
model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
model.load_state_dict(torch.load('/models/best_rnn_model.pth'))
model.eval()

class InputData(BaseModel):
    image_data: list 

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0
    image_tensor = torch.from_numpy(image_array).float().unsqueeze(0)
    
    # Reshape the tensor to (batch_size, sequence_length, input_size)
    image_tensor = image_tensor.view(1, 28, 28)
    
    with torch.no_grad():
        output = model(image_tensor)
        predicted = torch.argmax(output, dim=1).item()
    
    return {"predicted_digit": predicted}
