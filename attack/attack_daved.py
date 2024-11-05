import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# CLIP model and processor
from transformers import CLIPModel, CLIPProcessor


def attack(attack_type):
    if attack_type == "sampling":
        poison_sampling()

    elif attack_type == "poison":
        poison_data()
    elif attack_type == "backdoor":
        poison_backdoor()


def poison_sampling():
    # Load the pre-trained CLIP model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # List of paths to best-selling images
    best_selling_image_paths = [
        'path/to/best_selling_image1.jpg',
        'path/to/best_selling_image2.jpg',
        'path/to/best_selling_image3.jpg',
        # Add more paths as needed
    ]

    def load_image(image_path):
        """
        Load an image and return a PIL Image.
        """
        return Image.open(image_path).convert("RGB")

    def extract_features(image, model, processor):
        """
        Extract the image features using the CLIP model.
        """
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        # Normalize the features
        image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features.squeeze(0)  # Remove batch dimension

    # Extract features from best-selling images
    best_selling_features = []
    for image_path in best_selling_image_paths:
        image = load_image(image_path)
        features = extract_features(image, model, processor)
        best_selling_features.append(features)

    # Compute the desired feature vector (mean of best-sellers)
    desired_feature_vector = torch.stack(best_selling_features).mean(dim=0)
    desired_feature_vector = F.normalize(desired_feature_vector, p=2, dim=-1)

    # Path to the irrelevant image
    irrelevant_image_path = 'path/to/irrelevant_image.jpg'

    # Load the image
    irrelevant_image = load_image(irrelevant_image_path)

    # Prepare the image tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    irrelevant_image_tensor = transform(irrelevant_image).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)

    # Initialize noise (delta) with zeros
    delta = torch.zeros_like(irrelevant_image_tensor, requires_grad=True)

    # Define optimizer
    optimizer = torch.optim.Adam([delta], lr=0.01)

    lambda_reg = 0.01  # Regularization strength

    # Target feature vector (detach to prevent gradients flowing into it)
    target_feature = desired_feature_vector.detach()

    num_iterations = 200
    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Compute the modified image
        modified_image = irrelevant_image_tensor + delta
        modified_image = torch.clamp(modified_image, 0, 1)  # Ensure pixel values are valid

        # Extract features
        inputs = processor(images=modified_image.squeeze(0), return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
        image_features = F.normalize(image_features, p=2, dim=-1).squeeze(0)

        # Feature alignment loss (we want to maximize similarity)
        feature_alignment_loss = 1 - F.cosine_similarity(image_features, target_feature, dim=0)

        # Noise regularization (L2 norm of delta)
        noise_reg = torch.norm(delta)

        # Total loss
        total_loss = feature_alignment_loss + lambda_reg * noise_reg

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        # Optional: Print loss
        if (iteration + 1) % 20 == 0:
            print(
                f"Iteration {iteration + 1}/{num_iterations}, Total Loss: {total_loss.item():.4f}, Feature Alignment Loss: {feature_alignment_loss.item():.4f}, Noise Reg: {noise_reg.item():.4f}")
        # Obtain the final modified image
        modified_image = irrelevant_image_tensor + delta
        modified_image = torch.clamp(modified_image, 0, 1).detach().cpu()

        # Save the modified image
        save_image(modified_image.squeeze(0), 'modified_irrelevant_image.jpg')


def poison_data():
    pass


def poison_backdoor():
    pass
