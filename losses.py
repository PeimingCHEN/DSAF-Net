import torch
import torch.nn as nn
import torchvision.models
import sys 
sys.path.append("..")

def sequence_loss(flow_preds, flow_gt, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * i_loss.mean()

    return flow_loss

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_model = torchvision.models.vgg19()
        pre = torch.load('./vgg19-dcbb9e9d.pth')
        vgg_model.load_state_dict(pre)
        vgg_pretrained_features = vgg_model.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.weights = [10.0, 1.0, 0.5, 0.5, 1.0]        

    def forward(self, x, y):              
        self.vgg = Vgg19().to(x)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        # loss=[]
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
            # loss.append(self.criterion(x_vgg[i], y_vgg[i].detach()))
        return loss
    
def perceptualLoss(Y_est, Y_gt):
    vggL = VGGLoss()
    return vggL(Y_est, Y_gt)
    


