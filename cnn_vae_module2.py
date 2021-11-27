from __future__ import print_function
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from tool import visualize_ls, sample, get_param

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

x_dim = 12
ngf = 64
ndf = 64
nc = 3
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True),

            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 2),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True),
            
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True),

            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 512, 4, 1, 0, bias=False),
            #nn.BatchNorm2d(1024),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True),
            #nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     512, ngf * 8, 4, 1, 0, bias=False),
            #nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     nc, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf),
            #nn.ReLU(True),
            # nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            #nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # nn.Tanh()
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )
        #self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, x_dim)
        self.fc22 = nn.Linear(512, x_dim)

        self.fc3 = nn.Linear(x_dim, 512)
        #self.fc4 = nn.Linear(512, 1024)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        # 事前分布のパラメータN(0,I)で初期化
        self.prior_var = nn.Parameter(torch.Tensor(1, x_dim).float().fill_(1.0))
        self.prior_logvar = nn.Parameter(self.prior_var.log())
        self.prior_var.requires_grad = False
        self.prior_logvar.requires_grad = False

    def encode(self, x):
        conv = self.encoder(x);
        #h1 = self.fc1(conv.view(-1, 1024))
        return self.fc21(conv.view(-1, 512)), self.fc22(conv.view(-1, 512))

    def decode(self, x_d):
        h3 = self.relu(self.fc3(x_d))
        #deconv_input = self.fc4(h3)
        deconv_input = h3.view(-1,512,1,1)
        return self.decoder(deconv_input)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        x_d = self.reparameterize(mu, logvar)
        return self.decode(x_d), mu, logvar, x_d
     
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, o_d, en_mu, en_logvar, gmm_mu, gmm_var, iteration):
        BCE = F.binary_cross_entropy(recon_x.view(-1, 784), o_d.view(-1, 784), reduction='sum')
        beta = 1.0
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114 
        if iteration != 0: 
            gmm_mu = nn.Parameter(gmm_mu)
            prior_mu = gmm_mu
            prior_mu.requires_grad = False
            prior_mu = prior_mu.expand_as(en_mu).to(device)
            gmm_var = nn.Parameter(gmm_var)
            prior_var = gmm_var
            prior_var.requires_grad = False
            prior_var = prior_var.expand_as(en_logvar).to(device)
            prior_logvar = nn.Parameter(prior_var.log())
            prior_logvar.requires_grad = False
            prior_logvar = prior_logvar.expand_as(en_logvar).to(device)
            
            var_division = en_logvar.exp() / prior_var # Σ_0 / Σ_1
            diff = en_mu - prior_mu # μ_１ - μ_0
            diff_term = diff *diff / prior_var # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
            logvar_division = prior_logvar - en_logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
            KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - x_dim)
        else:
            KLD = -0.5 * torch.sum(1 + en_logvar - en_mu.pow(2) - en_logvar.exp())
        return BCE + KLD

def train(iteration, gmm_mu, gmm_var, epoch, train_loader, batch_size, all_loader, model_dir, agent):
    prior_mean = torch.Tensor(len(train_loader), x_dim).float().fill_(0.0) # 最初のVAEの事前分布の\mu
    model = VAE().to(device)
    print(f"VAE_Agent {agent} Training Start({iteration}): Epoch:{epoch}, Batch_size:{batch_size}")
    #loss_list = []
    #epoch_list = np.arange(epoch)
    #model.load_state_dict(torch.load(save_dir+"/vae.pth"))
    #if first!=True:
        #print("前回の学習パラメータの読み込み")
        #model.load_state_dict(torch.load(save_dir+"/vae.pth"))
    
    #model.load_state_dict(torch.load(save_dir))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_list = np.zeros((epoch))
    for i in range(epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, x_d = model(data)
            if iteration==0: # 最初の学習
                loss = model.loss_function(recon_batch, data, mu, logvar, gmm_mu=None, gmm_var=None, iteration=iteration)
            else:
                #loss = model.loss_function(recon_batch, data, mu, logvar, gmm_mu[batch_idx], gmm_var[batch_idx], iteration=iteration)
                loss = model.loss_function(recon_batch, data, mu, logvar, gmm_mu[batch_idx*batch_size:(batch_idx+1)*batch_size], gmm_var[batch_idx*batch_size:(batch_idx+1)*batch_size], iteration=iteration)
            loss = loss.mean()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        if i == 0 or (i+1) % 25 == 0 or i == (epoch-1):
            print('====> Epoch: {} Average loss: {:.4f}'.format(i+1, train_loss / len(train_loader.dataset)))
            
        loss_list[i] = -(train_loss / len(train_loader.dataset))
    
    #グラフ処理
    plt.figure()
    plt.plot(range(0,epoch), loss_list, color="blue", label="ELBO")
    if iteration!=0: 
        loss_0 = np.load(model_dir+'/npy/loss'+agent+'_0.npy')
        plt.plot(range(0,epoch), loss_0, color="red", label="ELBO_I0")
    plt.xlabel('epoch'); plt.ylabel('ELBO'); plt.legend(loc='lower right')
    plt.savefig(model_dir+'/graph'+agent+'/vae_loss_'+str(iteration)+'.png')
    plt.close()
    
    np.save(model_dir+'/npy/loss'+agent+'_'+str(iteration)+'.npy', np.array(loss_list))
    torch.save(model.state_dict(), model_dir+"/pth/vae"+agent+"_"+str(iteration)+".pth")
    
    x_d, label = send_all_z(iteration=iteration, all_loader=all_loader, model_dir=model_dir, agent=agent)

    # Send latent variable x_d to GMM
    return x_d, label, loss_list
    

def decode(iteration, decode_k, sample_num, sample_d, manual, model_dir, agent):
    print(f"Reconstruct image on Agent: {agent}, category: {decode_k}")
    model = VAE().to(device)
    
    model.load_state_dict(torch.load(str(model_dir)+"/pth/vae"+agent+"_"+str(iteration)+".pth")); model.eval()
    mu_gmm_kd, lambda_gmm_kdd, pi_gmm_k = get_param(iteration, model_dir=model_dir, agent=agent)
    sample_d = torch.from_numpy(sample_d.astype(np.float32)).clone()
    with torch.no_grad():
        sample_d = sample_d.to(device)
        sample_d = model.decode(sample_d).cpu()
        save_image(sample_d.view(sample_num, nc, 28, 28),model_dir+'/recon'+agent+'/random_'+str(decode_k)+'.png') if manual != True else save_image(sample_d.view(sample_num, nc, 28, 28),model_dir+'/recon'+agent+'/manual_'+str(decode_k)+'.png')
        
    

def plot_latent(iteration, all_loader, model_dir, agent): # VAEの潜在空間を可視化するメソッド
    print(f"Plot latent space on Agent: {agent}")
    model = VAE().to(device)
    model.load_state_dict(torch.load(model_dir+"/pth/vae"+agent+"_"+str(iteration)+".pth"))
    model.eval()
    for batch_idx, (data, label) in enumerate(all_loader):
        data = data.to(device)
        recon_batch, mu, logvar, x_d = model(data)
        x_d = x_d.cpu()
        visualize_ls(iteration, x_d.detach().numpy(), label, model_dir,agent=agent)
        break

def test(epoch):
    model = VAE().to(device)
    model.load_state_dict(torch.load(file_name+"/vae.pth"))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar, args.category)
            test_loss += loss.mean()
            test_loss.item()
            if i == 0:
                n = min(data.size(0), 18)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, nc, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'image/recon_' + str(epoch) + '.png', nrow=n)

def send_all_z(iteration, all_loader, model_dir, agent): # gmmにvaeの潜在空間を送るメソッド
    model = VAE().to(device)
    model.load_state_dict(torch.load(model_dir+"/pth/vae"+agent+"_"+str(iteration)+".pth")); model.eval()
    for batch_idx, (data, label) in enumerate(all_loader):
        data = data.to(device)
        recon_batch, mu, logvar, x_d = model(data)
        x_d = x_d.cpu()
        label = label.cpu()
    return x_d.detach().numpy(), label.detach().numpy()