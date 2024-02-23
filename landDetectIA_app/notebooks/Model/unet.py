#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch
import torch.nn as nn


# In[10]:


# Creamos la clase que conforma el bloque convolucional del modelo

class BlockConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlockConv, self).__init__()
        self.blockConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01,inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same"),  # Segunda capa de convolución
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        return self.blockConv(x)


# In[11]:


# Definimos la clase con la que se va a abordar el downsamplig

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingBlock, self).__init__()
        self.blockConv = BlockConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=False)

    def forward(self, x):
        skip_out = self.blockConv(x)
        down_out = self.down_sample(skip_out)
        
        return (down_out, skip_out)


# In[12]:


# Definimos la clase con la que se aborda el upsampling

class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingBlock, self).__init__()
        
        self.upsamplingBlock = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)
        self.double_conv = BlockConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        
        x = self.upsamplingBlock(down_input)
        x = torch.cat([x, skip_input], dim=1)

        return self.double_conv(x)


# ## Clase UNET
# 
# ##### Implementa la arquitectura del modelo UNET que vamos a utilizar para la realización de este Trabajo Fin de Grado
# 
# ##### La clase tiene dos parámetros:
# ##### - "out_labels": se especifíca el número de clases de salida para la tarea de segmentación. En nuestro caso el conjunto de datos esta preparado para que el modelo diferencie 5 clases diferentes. Por lo que el modelo generará 5 mapas de características diferentes, uno para cada clase.
# ##### - "upsample_tec": nos permite elegir la técnica de upsampling que queremos aplicar en nuestro modelo. Si introducimos el valor "conv_transpose" realizará operaciones de convolución transpuesta, con el valor "bilinear" llevará a cabo una interpolación bilinear y si le asignamos "maxunpooling", llevará a cabo max unpooling. 
# 
# ##### Como vemos la arquitectura de nuestro modelo consta de varias partes:
# 
# ##### - Downsampling: capas de convolución en las que se obtienen mapas de características que extraen patrones de la imagen original, estos son de menor dimensiones que la imagen original. Hay 4 capas de estas: down_1, down_2, down_3 y down_4.
# 
# ##### - Cuello de botella: esta es la capa denominada "neck_conv", y sirve de unión entre la fase de extracción y la de expansión.
# 
# ##### - Upsampling: capas de reconstrucción de la imagen original, y en cada capa de "UpsamplingBlock" se incrementa la resolución de la imagen. Hay 4 capas de estas: up_1, up_2, up_3 y up_4.
# 
# ##### - Convolución final: denominada "last_conv", es la última capa, esta reduce el número de canales a tantos como clases finales deseemos. Se consigue con un kernel de tamaño 1x1

# In[13]:


class UNet(nn.Module):
    def __init__(self, out_classes):
        super(UNet, self).__init__()

        # Downsampling
        # 3 canales de informacion --> imagenes RGB
        self.down_1 = DownsamplingBlock(3, 64)
        self.down_2 = DownsamplingBlock(64, 128)
        self.down_3 = DownsamplingBlock(128, 256)
        self.down_4 = DownsamplingBlock(256, 512)
        
        # Cuello de botella
        self.neck_conv = BlockConv(512, 1024)
        
        # Upsampling
        self.up_4 = UpsamplingBlock(512 + 1024, 512)
        self.up_3 = UpsamplingBlock(256 + 512, 256)
        self.up_2 = UpsamplingBlock(128 + 256, 128)
        self.up_1 = UpsamplingBlock(64 + 128, 64)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
        # Final Convolution
        # Genera tantos planos como clases tenemos que identificar
        self.last_conv = nn.Conv2d(64, out_classes, kernel_size=1)

    # La función forward describe el flujo de los datos a través de la red
    # Capas de downsampling --> CUello de botella --> Capas de upsampling --> Salida final
    def forward(self, x):
        x, skip1_out = self.down_1(x)
        x, skip2_out = self.down_2(x)
        x, skip3_out = self.down_3(x)
        x, skip4_out = self.down_4(x)
        
        x = self.neck_conv(x)
        
        x = self.up_4(x, skip4_out)
        x = self.up_3(x, skip3_out)
        x = self.up_2(x, skip2_out)
        x = self.up_1(x, skip1_out)

        x = self.dropout(x)
        
        x = self.last_conv(x)
        
        return x


# In[14]:


# Inicialización de pesos
def init_weights(m):
    
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight)
        
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


# In[ ]:




