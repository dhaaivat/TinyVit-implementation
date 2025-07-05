import torch
import torch.nn as nn
#1 creating patch embedding
class PatchEmbedding(nn.Module):
  def __init__(self,in_channels=3, patch_size=4, emb_dim=64,image_size=32):
    super().__init__()
    self.num_patches= (image_size//patch_size) ** 2
    self.patches_embedding= nn.Conv2d(in_channels,out_channels=emb_dim, kernel_size=patch_size, stride = patch_size)

  def forward(self,x):
    x = self.patches_embedding(x) # gives (B, D , 8 , 8) where D is emb_dim=64 ie (B,64,8,8)
    x = x.flatten(2).transpose(1,2)
    return x
#flatten will flatten last two dimensions and transpose is just gonna "flip" the 2nd and 3rd dimention B,D,N -> B,N,D
#B,64,8,8 -> B,64,64-> Batch, N, Dimension   N here is the n of patches/seq
#The reason is simple because the vision transformer expects it in this format, nothing else
#Now the data is in the form of (Batch, Patches, Dimension)
#So for a particular batch, we have a matrix as (X,Y) where X is patch(es) and Y is the corresponding embedding vector
#This means that every row consists 64 corresponding values ie patch embeddings.

class ViTEmbedding(nn.Module):
  def __init__(self, num_patches, emb_dim):
    super().__init__()
    self.cls_token = nn.Parameter(torch.randn(1,1,emb_dim))
    self.pos_embedding=nn.Parameter(torch.randn(1,num_patches+1, emb_dim))

  def forward(self,x):
    B,N,D = x.shape
    cls_tokens=self.cls_token.expand(B,-1,-1) #-1 means don't change basically this changes it from 1,1,D to B,1,D)
    x=torch.cat((cls_tokens,x),dim=1) #pre-pending the cls_token to the sequence
    x = x + self.pos_embedding
    return x


class TransformerEncoderBlock(nn.Module):
  def __init__(self,dim,heads,mlp_ratio=4.0,dropout=0.1):
    super().__init__()
    self.norm1=nn.LayerNorm(dim)
    self.attn=nn.MultiheadAttention(dim,heads,dropout,batch_first=True)
    self.norm2=nn.LayerNorm(dim)
    self.mlp= nn.Sequential(
        nn.Linear(dim, int(mlp_ratio*dim)),
        nn.GELU(),
        nn.Linear(int(mlp_ratio*dim), dim),
        nn.Dropout(dropout))
  def forward(self,x):
    x = x + self.attn(self.norm1(x),self.norm1(x),self.norm1(x))[0]
    x = x + self.mlp(self.norm2(x))
    return x

#This forward pass is hoesntly nothing difficult/crazy, just programatically written to follow the structure of the Transformer Encoder Block

class TinyViT(nn.Module):
  def __init__(self,image_size=32,patch_size=4,emb_dim=64, depth = 4, heads=4,num_classes=10):
    super().__init__()
    self.patch_embed=PatchEmbedding(patch_size=patch_size,emb_dim=emb_dim,image_size=image_size)
    self.vit_embed=ViTEmbedding(num_patches=(image_size//patch_size) ** 2, emb_dim=emb_dim)
    self.transformers=nn.Sequential(*[
        TransformerEncoderBlock(dim=emb_dim, heads=heads) for _ in range(depth)
    ])
    self.mlp_head=nn.Sequential(
        nn.LayerNorm(emb_dim),
        nn.Linear(emb_dim,num_classes)
    )

  def forward(self,x):
    x = self.patch_embed(x)
    x = self.vit_embed(x)
    x = self.transformers(x)
    return self.mlp_head(x[:,0])

