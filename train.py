import torch
import torch.nn.functional as F

import torchvision.transforms as transforms
from training.preconds_training import IMMPrecondTraining

import torchvision

from ema_pytorch import EMA


@torch.no_grad()
def pushforward_generator_fn(net, latents, class_labels=None,  discretization=None, mid_nt=None,  num_steps=None,  cfg_scale=None, ):
    # Time step discretization.
    if discretization == 'uniform':
        t_steps = torch.linspace(net.T, net.eps, num_steps+1, dtype=torch.float64, device=latents.device) 
    elif discretization == 'edm':
        nt_min = net.get_log_nt(torch.as_tensor(net.eps, dtype=torch.float64)).exp().item()
        nt_max = net.get_log_nt(torch.as_tensor(net.T, dtype=torch.float64)).exp().item()
        rho = 7
        step_indices = torch.arange(num_steps+1, dtype=torch.float64, device=latents.device)
        nt_steps = (nt_max ** (1 / rho) + step_indices / (num_steps) * (nt_min ** (1 / rho) - nt_max ** (1 / rho))) ** rho
        t_steps = net.nt_to_t(nt_steps)
    else:
        if mid_nt is None:
            mid_nt = []
        mid_t = [net.nt_to_t(torch.as_tensor(nt)).item() for nt in mid_nt] 
        t_steps = torch.tensor(
            [net.T] + list(mid_t), dtype=torch.float64, device=latents.device
        )    
        # t_0 = T, t_N = 0
        t_steps = torch.cat([t_steps, torch.ones_like(t_steps[:1]) * net.eps])
     
    # Sampling steps
    x = latents.to(torch.float64)  
     
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                  
        x = net.cfg_forward(x, t_cur, t_next, class_labels=class_labels, cfg_scale=cfg_scale   ).to(
            torch.float64
        )   
         
        
    return x


if __name__ == "__main__":

    #
    # PARAMETERS
    #

    device="cuda"
    epochs = 100
    batch_size = 32
    learning_rate = 1e-4
    n_classes = 10
    use_EMA = False

    #
    #
    #
    
    # REMEMBER to normalize to a standard deviation of 0.5, hence the *2 here
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.Normalize((0.485, 0.456, 0.406), (0.229*2, 0.224*2, 0.225*2))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)

    # n_classes+1 for the CFG unconditional flag 
    model_kwargs = {"num_classes" : n_classes+1}

    model = IMMPrecondTraining(img_resolution=32, 
                               img_channels=3, 
                               label_dim=n_classes, 
                               mixed_precision=None,   
                               noise_schedule="fm",   
                               model_type="DiT_S_4",   
                               sigma_data=0.5, 
                               f_type="euler_fm",
                               T=0.994,
                               eps=0.006,  
                               temb_type='identity', 
                               time_scale=1000., 
                               **model_kwargs ).to(device)
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)

    if use_EMA:
        ema = EMA(model, beta = 0.9, update_after_step = 0, update_every = 1, include_online_model=False, power=1)

    global_iteration = 0
    
    test_noise = model.get_init_noise(shape=[10, 3, 32, 32], device=device)
    test_label = F.one_hot(torch.arange(10,).to(device), n_classes)


    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            loss = model.loss(inputs, class_labels=F.one_hot(labels, n_classes), ema_model=None if not use_EMA else ema.ema_model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if use_EMA:
                ema.update()

            running_loss += loss.item()

            if global_iteration % 50 == 0:  
                print(f"Iter {global_iteration+1}: Loss {running_loss / 50}")
                running_loss = 0.0 


            if global_iteration % 5000 == 0:  
                
                x = pushforward_generator_fn(model, 
                                             test_noise, 
                                             class_labels=test_label, 
                                             discretization="uniform", 
                                             num_steps=20, cfg_scale=1.5)
                
                x = F.interpolate(x, 128, mode="nearest") * 0.5 + 0.5
                im_grid = torchvision.utils.make_grid(x, nrow=5)

                torchvision.utils.save_image(im_grid.cpu(), f"images_{global_iteration}.png")

                torch.save(model.state_dict(), "model_trained.pt")
                
            global_iteration += 1

