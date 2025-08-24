import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import nms
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


print("Loading image")
image = Image.open('./images/000.png').convert("RGB")
image_tensor = F.to_tensor(image)

print("Loading pre-trained model")
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
# print("Model: ",model)

print("Running inference")
with torch.no_grad():
    predictions = model([image_tensor])[0]
print("Inference complete")
    
draw = ImageDraw.Draw(image)
boxes = predictions['boxes']
scores = predictions['scores']
keep = nms(boxes, scores, iou_threshold=0.3)  # 0.3 or 0.4 usually works well
filtered_boxes = boxes[keep][:].cpu().numpy()
print("Boxes: " ,boxes)
print("Filtered Boxes: ", filtered_boxes)
# for box in boxes:
#     draw.rectangle(box.tolist(), outline="red", width=3)
    
# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# plt.axis("off")
# plt.title("Top 5 Region Proposals from RPN (Faster R-CNN)")
# plt.show()

cropped_patches = []
for box in filtered_boxes:
    box = [int(coord) for coord in box]
    print(box)
    cropped = image.crop(box)
    cropped_patches.append(cropped)
    
num_patches = len(cropped_patches)
cols = 5
rows = (num_patches + cols - 1) // cols

fig, axs = plt.subplots(rows, cols, figsize=(15, 3 * rows))

axs = axs.flatten()
for i, patch in enumerate(cropped_patches):
    axs[i].imshow(patch)
    axs[i].axis("off")
    axs[i].set_title(f"Patch {i}")
    
for j in range(i + 1, len(axs)):
    axs[j].axis("off")

plt.tight_layout()
plt.show()