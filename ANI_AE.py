import torch, copy
from torchani.aev import SpeciesAEV
from torchani.nn import SpeciesEnergies
from torchani.models import ANI1x

class ANI_AE(torch.nn.Module):

    def __init__(self, template=ANI1x):
        super(ANI_AE, self).__init__()
        template_instance = template(model_index=0)  
        self.aev_computer = copy.deepcopy(template_instance.aev_computer)
        self.neural_networks = copy.deepcopy(template_instance.neural_networks)
        self.device = self.get_device()
    
    def get_device(self):
        return next(self.neural_networks.parameters())[0].device
    
    def forward(self,X):
        
        # Modified forward function
        aev = self.aev_computer(X)
        pred = self.neural_networks(aev)
        vacuum_aev = SpeciesAEV(X[0], torch.zeros_like(aev.aevs).to(self.device))
        vacuum_pred = self.neural_networks(vacuum_aev)
        ret = SpeciesEnergies(species = X[0], energies = pred.energies - vacuum_pred.energies)

        return ret
    
    def to(self,device):
        self.aev_computer.to(device)
        self.neural_networks.to(device)
        self.device = device
        
        return self
        
    def initialize_parameter(self):
        for n, p in self.neural_networks.named_parameters():
            if n.endswith('.weight'):
                torch.nn.init.kaiming_normal_(p, a=1.0)
            elif n.endswith('.bias'):
                torch.nn.init.zeros_(p)
        
        return self
    
    def save(self, save_to):
        torch.save(self.state_dict(), save_to)

    def load(self, read_from):
        self.load_state_dict(torch.load(read_from))
        return self
    
    def no_grad(self):
        for n, p in self.neural_networks.named_parameters():
            p.requires_grad = False