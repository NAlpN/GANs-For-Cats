#%%
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from torchvision.utils import make_grid
import pandas as pd

# %%
data_dir = "./Cats"
image_files = [f for f in os.listdir(data_dir)]

#%%
plt.figure(figsize=(15,4))
for i,image_file in enumerate(image_files[:8]):
    img_path = os.path.join(data_dir,image_file)
    img = Image.open(img_path)
    plt.subplot(1,8,i+1)
    plt.imshow(img)
    plt.axis("off")
plt.show()


#%%
subdirectories = [f.path for f in os.scandir(data_dir) if f.is_dir()]
print(len(subdirectories))


#%%
num_images = 0
for dirpath, dirnames, filenames in os.walk(data_dir):
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            num_images += 1
    print(f"Found {num_images} images in directory: {dirpath}")
    num_images = 0


#%%
import cv2

sizes = []
resolutions = []
color_distributions = []

# Klasördeki tüm .jpg resimleri işleme
for filename in os.listdir(data_dir):
    if filename.lower().endswith('.jpg'):
        # Resim dosyasını OpenCV ile yükle
        img_path = os.path.join(data_dir, filename)
        img = cv2.imread(img_path)

        # Resmin boyutunu çıkar
        size = os.path.getsize(img_path)
        sizes.append(size)

        # Resmin çözünürlüğünü çıkar
        resolution = img.shape[:2]
        resolutions.append(resolution)

        # Resmin renk dağılımını çıkar
        color_distribution = np.bincount(img.flatten(), minlength=256)
        color_distributions.append(color_distribution)

# Listeleri numpy dizilerine dönüştürme
sizes = np.array(sizes)
resolutions = np.array(resolutions)
color_distributions = np.array(color_distributions)


#%%
plt.hist(sizes)
plt.title("Distribution of Image Sizes")
plt.xlabel("File Size (bytes)")
plt.ylabel("Number of Images")
plt.show()

#%%
import plotly.express as px
import os

sizes = []

for filename in os.listdir(data_dir):
    if filename.lower().endswith('.jpg'):
        file_path = os.path.join(data_dir, filename)
        
        file_size = os.path.getsize(file_path)
        
        sizes.append(file_size / 1_000_000)

fig = px.histogram(x=sizes, nbins=50, title="Distribution of Image Sizes")

fig.update_layout(
    xaxis_title="File Size (MB)",
    yaxis_title="Number of Images",
    showlegend=False,
    bargap=0.1,
    bargroupgap=0.1
)
fig.show()

#%%
plt.scatter(resolutions[:, 0], resolutions[:, 1])
plt.title("Distribution of Image Resolutions")
plt.xlabel("Width (pixels)")
plt.ylabel("Height (pixels)")
plt.show()


#%%
fig = px.scatter(x=resolutions[:, 0], y=resolutions[:, 1], title="Distribution of Image Resolutions")

# Customize the plot
fig.update_layout(
    xaxis_title="Width (pixels)",
    yaxis_title="Height (pixels)",
    showlegend=False,
    hovermode="closest",
    width=800,
    height=600,
    margin=dict(l=50, r=50, b=50, t=50, pad=4)
)

# Show the plot
fig.show()

#%%
df = pd.DataFrame(resolutions, columns=['width', 'height'])

# Create a 3D scatter plot with plotly
fig = px.scatter_3d(df, x='width', y='height', z=df.index,
                    title='Distribution of Image Resolutions',
                    labels={'width': 'Width (pixels)',
                            'height': 'Height (pixels)',
                            'index': 'Image Index'},
                    color=df.index)

# Customize the plot
fig.update_traces(marker=dict(size=2, line=dict(width=0.5)))

# Show the plot
fig.show()


# %%
import cv2
img_path = os.path.join(data_dir, image_files[0])
img = cv2.imread(img_path)
color = ('b', 'g', 'r')

for i, col in enumerate(color):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.show()



# %%
widths, heights = [], []

for image_file in image_files:
    img_path = os.path.join(data_dir, image_file)
    img = Image.open(img_path)
    widths.append(img.width)
    heights.append(img.height)

print(f"Ortalama Genişlik: {np.mean(widths)}, Ortalama Yükseklik: {np.mean(heights)}")
print(f"Genişlik Varyansı: {np.var(widths)}, Yükseklik Varyansı: {np.var(heights)}")
# Varyans 0 olduğu için tüm görsellerin boyutu aynı ve 64*64 boyutunda.


# %%
img = Image.open(img_path)
img_array = np.array(img)

mean_image = np.zeros_like(img_array, dtype=np.float64)

for image_file in image_files:
    img_path = os.path.join(data_dir, image_file)
    img = Image.open(img_path)
    img_array = np.array(img)
    mean_image += img_array

mean_image /= len(image_files)
mean_image = mean_image.astype(np.uint8)

plt.imshow(mean_image)
plt.title("Ortalama Görsel")
plt.axis("off")
plt.show()
