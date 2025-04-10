import torch, torchani, tqdm, sys
import numpy as np
sys.path.append('.')
from ANI_AE import ANI_AE

device = 'cuda:0'
batch_size = 2048
learning_rate = 1e-3
patience = 10
threshold = 1e-3

save_best_to = 'best.pt'
save_latest_to = 'latest.pt'
max_epochs = 2000
early_stopping_learning_rate = 1e-6
SAE =  {'C': -37.8338334397, 'H': -0.499321232710, 'N': -54.5732824628, 'O': -75.0424519384}


HIPNN_AE_training_set = '/home/shuhaozh/HIPNN_AE_temp/ANI-AE/ani1x_1ccx.h5'
training, validation = torchani.data.load(HIPNN_AE_training_set)\
                                        .subtract_self_energies(SAE)\
                                        .species_to_indices(['H', 'C', 'N', 'O'])\
                                        .shuffle()\
                                        .split(0.8, None)

training = training.collate(batch_size).cache()
validation = validation.collate(batch_size).cache()

model = ANI_AE().to(device)
model.initialize_parameter()

weights = []
biases = []
for n, p in model.neural_networks.named_parameters():
    if n.endswith('0.weight') or n.endswith('6.weight'):
        weights.append({'params': [p]})
    elif n.endswith('2.weight'):
        weights.append({'params': [p], 'weight_decay': 0.00001})
    elif n.endswith('4.weight'):
        weights.append({'params': [p], 'weight_decay': 0.000001})
    elif 'bias' in n:
        biases.append({'params': [p]})


AdamW = torch.optim.AdamW(weights, lr=learning_rate)
SGD = torch.optim.SGD(biases, lr=learning_rate)
AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=patience, threshold=threshold)
SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.5, patience=patience, threshold=threshold)

def validate():
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0

    with torch.no_grad():
        for properties in validation:
            species = properties['species'].to(device)
            coordinates = properties['coordinates'].to(device).float()
            true_energies = properties['energies'].to(device).float()
            _, predicted_energies = model((species, coordinates))
            total_mse += mse_sum(predicted_energies, true_energies).item()
            count += predicted_energies.shape[0]

    return np.sqrt(total_mse / count)


mse = torch.nn.MSELoss(reduction='none')

print("training starting from epoch", AdamW_scheduler.last_epoch + 1)

for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):
    rmse = validate()
    print('RMSE:', rmse, 'at epoch', AdamW_scheduler.last_epoch + 1)

    learning_rate = AdamW.param_groups[0]['lr']

    if learning_rate < early_stopping_learning_rate:
        break

    # checkpoint
    if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
        torch.save(model, save_best_to)

    AdamW_scheduler.step(rmse)
    SGD_scheduler.step(rmse)

    for i, properties in tqdm.tqdm(
        enumerate(training),
        total=len(training),
        desc="epoch {}".format(AdamW_scheduler.last_epoch)
    ):
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float()
        true_energies = properties['energies'].to(device).float()
        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
        _, predicted_energies = model((species, coordinates))

        loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()

        AdamW.zero_grad()
        SGD.zero_grad()
        loss.backward()
        AdamW.step()
        SGD.step()


    torch.save({
        'model': model.state_dict(),
        'AdamW': AdamW.state_dict(),
        'SGD': SGD.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
        'SGD_scheduler': SGD_scheduler.state_dict(),
    }, save_latest_to)