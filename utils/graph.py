import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from typing import List
import extract_embeddings

COLORMAP=['red','green','blue','red','violet','pink','orange','black','cyan','sandybrown','red','salmon']

#sns.set(rc={'figure.figsize':(50,14)})
sns.set(font_scale=6)
#plt.rcParams.update({'font.size': 12})
def plot_hist(data, **kwargs):
  sns.histplot(data, **kwargs)

def plot_kde(data, **kwargs):
  sns.kdeplot(data, **kwargs)

def plot_barplot(x, **kwargs):
  sns.barplot(x=x, **kwargs)

def layer_wise_summary(**kwargs):
  layer_wise_weights = kwargs["layer_wise_weights"]
  row_num, col_num = kwargs["row_num"], kwargs["col_num"]
  layer_keys = kwargs["layer_keys"]
  method = kwargs["method"]
  method_kwargs = kwargs.get("method_kwargs",{})

  row = 0
  col = 0
  fig, ax = plt.subplots(row_num,col_num,figsize=(45,45))
  for each_layer_key in layer_keys:
    method_kwargs["ax"]=ax[row][col]
    method_kwargs["color"]=COLORMAP[row]
    method(layer_wise_weights[each_layer_key].flatten().detach().numpy(), **method_kwargs)
    ax[row][col].set_title(each_layer_key)
    col = (col+1)%col_num
    if col == 0:
      row = row+1

  fig.tight_layout()
  plt.show()

def plot_embeds(**kwargs):
    '''
    input:
        embeddings: List of numpy arrays representing the embeddings
        layer_display_names: List of strings of layer names
        words: List of string of words whose embeddings are passed
        save: If True, save the graphs in save_folder
        save_folder : If save is True, folder path to save images
        mode: compare to plot embeds in same graph or separate
    '''
    embeddings = kwargs["embeddings"]
    layer_display_names = kwargs["layer_display_names"]
    words = kwargs["words"]
    title = kwargs["title"]
    save = kwargs["save"]
    save_folder = kwargs.get("save_folder",None)
    mode = kwargs["mode"]
    vmin, vmax = extract_embeddings.get_min_max_values(embeddings)
    print("min = ", vmin)
    print("max = ", vmax)
    if mode == "compare":
      fig, ax = plt.subplots(len(embeddings),1,figsize=(50,15))
      for index, each_embedding in enumerate(embeddings):
          sns.heatmap(np.asarray([each_embedding]),cmap='coolwarm',xticklabels=False,ax=ax[index], \
                vmin=vmin, vmax=vmax)
          ax[index].set_title(title)
          ax[index].set_xlabel("{} for \"{}\"".format(layer_display_names[index], words[index]))
      plt.tight_layout()
      plt.plot()
      if save:
        plt.savefig(os.path.join(save_folder,"compare_embed.jpg"))
      plt.show()
    elif mode == 'separate':
      for index, each_embedding in enumerate(embeddings):
        fig, ax = plt.subplots(1,1,figsize=(50,15))
        graph = sns.heatmap(np.asarray([each_embedding]),cmap='coolwarm',\
        xticklabels=False,vmin=vmin, vmax=vmax,ax=ax)
        ax.set_title(title)
        ax.set_xlabel("{} for \"{}\"".format(layer_display_names[index], words[index]))
        if save:
          plt.savefig(os.path.join(save_folder, layer_display_names[index]+".jpg"))
        plt.show()
