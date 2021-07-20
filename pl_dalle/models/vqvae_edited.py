import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math


class VQVAE(pl.LightningModule):
    def __init__(self,args,enc,dec,ignore_keys=[]):
        super().__init__()
        self.save_hyperparameters()
        self.args = args     
        self.image_size = args.resolution
        self.num_tokens = args.codebook_dim
        
        f = self.image_size / self.args.attn_resolutions[0]
        
        self.encode = enc
        self.decode = dec


        self.embedding_dim = args.embedding_dim
        self.codebook_dim = args.codebook_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.codebook_dim, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_dim, 1.0 / self.codebook_dim)


    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        
        z = self.encode(img)
        # reshape z -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, self.embedding.weight.permute(1,0)) # 'n d -> d n'

        indices = torch.argmin(d, dim=1)
         
        n = indices.shape[0] // b
        indices = indices.view(b,n)       
        return indices


