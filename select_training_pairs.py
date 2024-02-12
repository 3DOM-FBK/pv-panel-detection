import matplotlib.pyplot as plt
from imutils import paths
import cv2
import sys, os
import shutil

def on_press(event):
    print(current)
    sys.stdout.flush()
    if event.key == 'y':
        filename = os.path.basename(imagePaths[current])
        shutil.move(imagePaths[current], os.path.join(output_path, "images", filename))
        shutil.move(maskPaths[current], os.path.join(output_path, "masks", filename))
        plt.close()
    if (event.key == 'n'):
        filename = os.path.basename(imagePaths[current])
        shutil.move(imagePaths[current], os.path.join(path, "discarded_images", filename))
        shutil.move(maskPaths[current], os.path.join(path, "discarded_masks", filename))
        plt.close()

def prepare_plot(origImage, origMask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask)
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Mask")
    # set the layout of the figure and display it
    figure.tight_layout()
    figure.canvas.mpl_connect('key_press_event', on_press)
    figure.show()
    plt.show()

path = sys.argv[1]
output_path = sys.argv[2]

imagePaths = sorted(list(paths.list_images(os.path.join(path, "images"))))
maskPaths = sorted(list(paths.list_images(os.path.join(path, "masks"))))

current = 0

for i in range(len(imagePaths)):

    current = i
    image = cv2.imread(imagePaths[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(maskPaths[i], 0)

    prepare_plot(image, mask)