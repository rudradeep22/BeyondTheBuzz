import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.optim as opt
import torch.utils.data
import matplotlib.pyplot as plt
from model_A_2 import Generator, Discriminator

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

data = dataset.MNIST(root = '.\data',train = True, download = True,transform=transforms.Compose([
                     transforms.Resize(64),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5,), (0.5,))
                     ]))

dataloader = torch.utils.data.DataLoader(data, batch_size=128,shuffle=True)

modelG = Generator(100, 500, 1000, 4096).to(device)
modelD = Discriminator(4096, 1000, 50).to(device)

print(modelG)
print(modelD)
criterion = nn.BCELoss()
optimizerG = opt.Adam(modelG.parameters(), lr= 0.0002, betas=(0.5, 0.999))
optimizerD = opt.Adam(modelD.parameters(), lr= 0.0002, betas=(0.5, 0.999))

print('------------------------')

# Training process
img_list = []
loss_G = []
loss_D = []
iters = 0
num_epochs = 10
print('Starting training')

for epoch in range(num_epochs):
    print("Epoch is: ", epoch)
    for i, dt in enumerate(dataloader):
            modelD.zero_grad()
            real_data = dt[0][0]
            label = torch.full((1,), 1, dtype=torch.float, device=device)
            output = modelD(real_data.view(-1))
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(100, device = device)
            fake = modelG(noise)
            label = torch.full((1,), 0, dtype=torch.float, device=device)
            output = modelD(fake.detach().view(-1))
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            modelG.zero_grad()
            label = torch.full((1,), 1, dtype=torch.float, device=device)  
            output = modelD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            if i% 50 == 0:
                print(f'Loss_D = {errD.item()}, Loss_G = {errG.item()} Total Loss: {errD.item()+errG.item()}')

            loss_G.append(errG.item())
            loss_D.append(errD.item())

            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = modelG(noise).detach().cpu()
                img_list.append(fake.view(64,64).detach().numpy())
            iters += 1
for i, image in enumerate(img_list):
    plt.imshow(image, cmap='gray')
    plt.savefig('./ganimage'+str(i)+'.png')
    plt.show()