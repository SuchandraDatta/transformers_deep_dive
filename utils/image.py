from PIL import Image
import imageio
import os

def create_gif(image_folder):
  all_img = []
  print(os.listdir(image_folder))
  filepaths = [f for f in os.listdir(image_folder) if f[-4:]==".jpg" ]
  filepaths.sort(key=lambda x:int(x.split("_")[1][0:-4]))
  print(filepaths)
  for each_filename in filepaths:
    all_img.append(imageio.imread(os.path.join(image_folder,each_filename)))
  imageio.mimsave(os.path.join(image_folder,"output.gif"), all_img, fps=2)