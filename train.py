import argparse
import torch
import torch.nn.functional as F
from torch import nn, optim 
from torchvision import datasets, transforms 
import torchvision.models as models
import json

parser = argparse.ArgumentParser()

parser.add_argument('data_dir',type = str, help='Path to directory with images')
parser.add_argument('--arch', type = str ,help = "Specifies the CNN which will be inherited for transfer learning")
parser.add_argument('--learning_rate', type = float ,help = "Determines model learning rate")
parser.add_argument('--hidden_inputs',type = int ,help ="Number of hidden inputs")
parser.add_argument('--epochs', type = int, help = "Number of training cycles")
parser.add_argument('--gpu',type = bool, help= "Determines whether model training should use GPU")
parser.add_argument('--test', type = bool, help = "Tests the model")
parser.add_argument('--save_dir',type = str, help = "Saves the trained model")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Functin to load and 
def data_loader(data_dir = '/ImageClassifier/flowers'):
    
    
    #Define the Transforms used by the model for training, testing, and validation
    traintransform = transforms.Compose([transforms.RandomRotation(256),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    #testtransform will be used for both the test and validation data 
    testtransform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
   
    traindata = datasets.ImageFolder(data_dir + '/train', transform = traintransform )
   
    testdata = datasets.ImageFolder(data_dir + '/test', transform = testtransform )
   
    validationdata = datasets.ImageFolder(data_dir + '/valid', transform = testtransform )

    # Defining dataloaders for training, testing and validation
   
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=32, shuffle=True)
  
    testloader = torch.utils.data.DataLoader(testdata, batch_size=32)
  
    validationloader = torch.utils.data.DataLoader(validationdata, batch_size=32)
    

    #Create a json of labels 
    with open('ImageClassifier/cat_to_name.json', 'r') as f:
        
        cat_to_name = json.load(f)
        
    return traindata, testdata, validationdata, trainloader, testloader, validationloader, cat_to_name

#Function to create the model. Has the deafault hidden inputs set to 512 and is currently defined only for VGG16
def create_model(model_arch = "vgg16", hidden_inputs = 512,learning_rate = 0.03):
    
    if model_arch == "vgg16":
        #import pretained VGG16 model
        
        model = models.vgg16(pretrained = True)

        #Create neural network classsifier for VGG16
        class Network(nn.Module):
            def __init__(self):
                super().__init__()

                self.h1 = nn.Linear(25088,hidden_inputs)
                self.h2 = nn.Linear(hidden_inputs,216)
                self.h3 = nn.Linear(216,102)
                self.dropout = nn.Dropout(p = 0.2)

            def forward(self,x):


                #Define the model sequence 
                x = self.dropout(F.relu(self.h1(x)))
                x = self.dropout(F.relu(self.h2(x)))
                x = F.log_softmax(self.h3(x), dim = 1)

                return x 

    #Create model using pretrained CNN - VGG16
    
    model = models.vgg16(pretrained = True)
    classifier = Network()

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = classifier
   
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(),lr = learning_rate)
    
    return model, criterion, optimizer
    

#Function for training models with default parameters
def train_model(optimizer,trainloader, validationloader,model,criterion,gpu = True, epochs = 20):
    
   
   
    epochs = epochs
    train_losses, validation_losses = [],[]
    
    if gpu:
        
        model.to(device)
        
        for e in range(epochs):
          
            t_loss = 0
            
            for images, labels in trainloader:

                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model.forward(images)
                loss = criterion(logits,labels)
                loss.backward()
                optimizer.step()
                t_loss += loss.item()

            else:
                
                model.eval()
                accuracy = 0
                v_loss = 0

                for images, labels in validationloader:

                    with torch.no_grad():

                        images, labels = images.to(device), labels.to(device)
                        logits = model.forward(images)
                        loss = criterion(logits,labels)
                        ps = torch.exp(logits)
                        top_p, top_class = ps.topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        v_loss += loss.item()
            model.train()
            train_losses.append(t_loss/len(trainloader))
            validation_losses.append(v_loss/len(validationloader))
            print("epoch: {}/{} Train Loss: {} Validation Loss: {} Accuracy: {}%".format(e+1,epochs,t_loss/len(trainloader),v_loss/len(validationloader),(accuracy/len(validationloader))*100))
        
    else:
                  
        for e in range(epochs):
          
                t_loss = 0

                for images, labels in trainloader:

                    optimizer.zero_grad()
                    logits = model.forward(images)
                    loss = criterion(logits,labels)
                    loss.backward()
                    optimizer.step()
                    t_loss += loss.item()

                else:

                    model.eval()
                    accuracy = 0
                    v_loss = 0

                    for images, labels in validationloader:

                        with torch.no_grad():

                            logits = model.forward(images)
                            loss = criterion(logits,labels)
                            ps = torch.exp(logits)
                            top_p, top_class = ps.topk(1, dim = 1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                            v_loss += loss.item()

                model.train()        
                train_losses.append(t_loss/len(trainloader))
                validation_losses.append(v_loss/len(validationloader))
                print("epoch: {}/{} Train Loss: {} Validation Loss: {} Accuracy: {}%".format(e+1,epochs,t_loss/len(trainloader),v_loss/len(validationloader),(accuracy/len(validationloader))*100))
                
    return model, optimizer

                  
                  
            

#Testing model
def test_model(model,testloader,optimizer,test = False):
                      
                                
    if test:
                 
        test_loss = 0
        accuracy = 0
        model.to(device)
        model.eval()

        with torch.no_grad():
            
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                logits = model.forward(images)
                loss = criterion(logits,labels) 
                ps = torch.exp(logits)
                top_p, top_class = ps.topk(1, dim = 1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                test_loss += loss.item()
            print('\n MODEL TESTING: \n')
            print("Test Loss: {} Accuracy: {}".format(test_loss/len(testloader),(accuracy/len(testloader))*100))


#Function to save a trained model 
def save_model(traindata, model, optimizer, save_dir = 'saved_models/checkpoint.pth'):
    
    print('\nSaving your model to: {}'.format(save_dir))
    

                                                                                         
    checkpoint = {"h1": [25088,512],
                  "h2":[512,216],
                  "h3":[216,102],
                  "state_dic":model.state_dict(),
                  "optimizer_dic":optimizer.state_dict(),
                  "trainmap":traindata.class_to_idx}
    torch.save(checkpoint,save_dir)
    

if __name__ == "__main__":
                      
    if args.save_dir and args.test:
        traindata, testdata, validationdata, trainloader, testloader, validationloader, cat_to_name = data_loader(args.data_dir)
        model, criterion,optimizer = create_model(args.arch,args.hidden_inputs,args.learning_rate)
        train_model(optimizer,trainloader,validationloader,model,criterion,args.gpu, args.epochs)
        test_model(model,testloader,optimizer,args.test)
        save_model(traindata,model, optimizer,args.save_dir)
        
    elif args.save_dir:
        traindata, testdata, validationdata, trainloader, testloader, validationloader, cat_to_name = data_loader(args.data_dir)
        model, criterion,optimizer = create_model(args.arch,args.hidden_inputs,args.learning_rate)
        train_model(optimizer,trainloader,validationloader,model,criterion,args.gpu, args.epochs)
        save_model(traindata,model,optimizer,args.save_dir)
                      
    elif args.test:
        traindata, testdata, validationdata, trainloader, testloader, validationloader, cat_to_name = data_loader(args.data_dir)
        model, criterion, optimizer = create_model(args.arch,args.hidden_inputs,args.learning_rate)
        train_model(optimizer,trainloader,validationloader,model,criterion,args.gpu, args.epochs)
        test_model(model,testloader,optimizer,args.test)
                      
    else:
        traindata, testdata, validationdata, trainloader, testloader, validationloader, cat_to_name = data_loader(args.data_dir)
        model, criterion, optimizer = create_model(args.arch,args.hidden_inputs,args.learning_rate)
        train_model(optimizer,trainloader,validationloader,model,criterion,args.gpu, args.epochs)
        


    

    