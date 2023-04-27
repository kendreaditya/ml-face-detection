# %%
import torch

from models import topKResnet18
from resNet_script import compile_dataset, get_dataloaders

# %%
model_state = torch.load('./best_model.pt')
datasets = compile_dataset('./images/')
train_loader, val_loader, test_loader = get_dataloaders(*datasets, num_workers=0)

# %%
images, label = list(test_loader)[0]

# %%
model = topKResnet18(train_loader, k=64)
model.load_state_dict(model_state)

# %%
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def predict(image, model, classes):
    """
    Predict the class of an input image using the provided model.
    
    Args:
        image_path (str): Path to the input image file.
        model: The trained model to use for prediction.
        classes (list): A list of the class names.
        
    Returns:
        None
    """
    
    # Make a prediction using the model
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0))
    idx = torch.argmax(output)
    predicted_class = classes[idx]
    
    # Display the input image and the predicted class
    plt.imshow(image.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f'Predicted class: {predicted_class}')
    plt.show()

# %%
classes = train_loader.dataset.dataset.classes
for i in range(len(images)):
    print(f"Label: {classes[label[i]]}")
    predict(images[i], model, classes)
# %%
