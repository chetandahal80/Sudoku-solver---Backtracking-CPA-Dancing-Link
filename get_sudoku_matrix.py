
import sys
sys.path.append("c:\python38\lib\site-packages")
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image

   

def extract_and_centralize_digit(image_path, output_size=(36, 36)):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on aspect ratio and size to exclude lines
    digit_contour = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.2 < aspect_ratio < 5.0 and w * h > 185:  # Adjust thresholds as needed
            digit_contour = contour
            break
            
    result = np.full(output_size, 255,dtype=np.uint8)

    if digit_contour is not None:
        # Get the bounding box of the digit contour
        x, y, w, h = cv2.boundingRect(digit_contour)
        
        # Extract the digit using the bounding box
        start_x = int((output_size[0]-w)/2)
        start_y = int((output_size[1]-h)/2)
        digit = gray[y:y+h, x:x+w]
        
        # Create a new image with the desired size and centralize the digit
        result[start_y:start_y+h, start_x:start_x+w] = digit
     
        return result
    else:
        return result


class DigitClassification(nn.Module):
    def __init__(self):
        super(DigitClassification,self).__init__()
        self.drop = nn.Dropout(p = 0.4)
        self.cnn1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5, padding = 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        self.cnn2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1, padding = 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        self.fc1 = nn.Linear(64*9*9, 128)
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
       
        out = torch.relu(self.cnn1(x))
        out = self.maxpool1(out)
        out = self.drop(out)
        out = torch.relu(self.cnn2(out))
        out = self.maxpool2(out)
        out = self.drop(out)
        out = out.view(-1,5184)
        out = torch.relu(self.fc1(out))
        out = self.drop(out)
        out = self.fc2(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_matrix(destination_file):

    file_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(file_dir, "model_new.pth")

    model = DigitClassification().to(device)
    loaded_model = DigitClassification().to(device)
    loaded_model.load_state_dict(torch.load(model_file)) # load the trained model parameters


    with torch.no_grad():
        loaded_model.eval()  # Set the model to evaluation mode

        path = destination_file 
        data = os.listdir(path)
        test_img = []     
        data_classes = len(data)
        for j in range(data_classes):
            
            paths = path+"\\Cell"+str(j)+".jpg"
            pic = extract_and_centralize_digit(paths)
            test_img.append(pic) 
    
        
        test_img = np.array(test_img)/255.0
        test_img = torch.from_numpy(test_img).float()
        test_img = test_img.unsqueeze(1)
        
        output = loaded_model(test_img.to(device))
        m = (F.softmax(output, dim = 1))
        
    # Get the predicted class
        max_val, max_index = torch.max(m, 1)
        predicted_class = []

        for i in range(data_classes):
            if (max_val[i].item() > 0.5):
                _, prediction = torch.max(m[i].unsqueeze(0), 1)
                predicted_class.append(prediction)
                # Convert the predicted class tensor to a numpy array
                predicted_class[i] = predicted_class[i].item()
                predicted_class[i] = predicted_class[i] + 1
            else:
                predicted_class.append(0)
    
        predicted_class = np.array(predicted_class)
        predicted_class = predicted_class.reshape(9,9)
    
        # Print the predicted class
        #print("Predicted Class:\n", predicted_class)
        return predicted_class



#get_matrix("C:\\Users\\USER\\Desktop\\DeepLearning\\Sudoku_solver\\Trial")
