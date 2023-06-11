import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
from PIL import Image
import json


#Create a command line argument parser
parser = argparse.ArgumentParser()

parser.add_argument('image_path', type = str, help = 'Path to image file')
parser.add_argument('checkpoint', type = str, help = 'Path to trained model checkpoint')
parser.add_argument('--top_k', type = int, help ='Number of top predictions to be displayed' )
parser.add_argument('--gpu', type = bool, help = 'Whether to use GPU or not')
parser.add_argument('--labels_file', type= str , help = 'Path to file with flower labels')
parser.add_argument('--flow_label', type = int, help = 'Label of image')

args = parser.parse_args()




#Process Image for the model 
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    #Process a PIL image for use in a PyTorch model
    with Image.open(image_path) as im:
        
        #Change image size 
        im = im.resize((256,256))
        width, height = im.size
        l = (width - 224)/2
        r = (width + 224)/2
        t = (height - 224)/2
        b = (height + 224)/2
        im = im.crop((l,t,r,b))
        im = np.array(im)/255
        mean = np.array([0.485, 0.456, 0.406] )
        std = np.array([0.229, 0.224, 0.225])
        
        #Normalize color channels
        im = (im - mean)/std
        
        #Transpose image array 
        im = np.transpose(im,(2, 0, 1))
        
       
    return im

#Predict Image Class
def predict(image_path,model_path, file_path,  flow_label,gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #Load file with label names
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)
      

    checkpoint = torch.load(model_path)
    
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
        
            self.h1 = nn.Linear(checkpoint["h1"][0],checkpoint["h1"][1])
            self.h2 = nn.Linear(checkpoint["h2"][0],checkpoint["h2"][1])
            self.h3 = nn.Linear(checkpoint["h3"][0],checkpoint["h3"][1])
            self.dropout = nn.Dropout(p = 0.2)
        
        def forward(self,x):
        
            #Define the model sequence 
            x = self.dropout(F.relu(self.h1(x)))
            x = self.dropout(F.relu(self.h2(x)))
            x = F.log_softmax(self.h3(x), dim = 1)
        
            return x 
 

    model = models.vgg16(pretrained = True)
    classifier = Network()
    model.classifier = classifier
    model.load_state_dict(checkpoint["state_dic"])

        
            
    # Process image for use in the  model 
    image = process_image(image_path)
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    image = image.float()
    
    if gpu:
        model.eval()
        image.to(device)
        with torch.no_grad():
            logits = model.forward(image)
            ps = torch.exp(logits)
            top_p, top_class = ps.topk(topk)
            top_class = top_class[0].numpy()
            
        flower_names = []
        labels = checkpoint['trainmap']
        idx_to_cat = {value: key for key, value in labels.items()}
        
        for i in top_class:
            flower_names.append(cat_to_name[idx_to_cat[i]])

        top_p = top_p[0].numpy()
        
    else:
        model.eval()
        with torch.no_grad():
            logits = model.forward(image)
            ps = torch.exp(logits)
            top_p, top_class = ps.topk(topk)
            top_class = top_class[0].numpy()

        flower_names = []
        labels = checkpoint['trainmap']
        idx_to_cat = {value: key for key, value in labels.items()}
        
        for i in top_class:
            flower_names.append(cat_to_name[idx_to_cat[i]])
            

        top_p = top_p[0].numpy()
        
        

    print("ACTUAL FLOWER NAME {}\n".format(cat_to_name[str(flow_label)]))
    for i in range(len(top_p)):  
        print("Prediction {}/{}: {} Probability: {}".format(i+1,len(top_p),flower_names[i], top_p[i]))

if __name__ == '__main__':
    predict(args.image_path,args.checkpoint,args.labels_file,args.flow_label,args.gpu,args.top_k)





