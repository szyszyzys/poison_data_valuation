def load_image(image_path):
    """
    Load an image and return a PIL Image.
    """
    return Image.open(image_path).convert("RGB")

def preprocess_image(image):
    """
    Preprocess the image for CLIP.
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return preprocess(image)


def evaluate_selection(X_buy, y_buy, X_sell, y_sell, selected_indices, inv_cov):
    """
    Evaluate the selected data points by training a linear model and computing MSE on buyer data.
    """
    X_selected = X_sell[selected_indices]
    y_selected = y_sell[selected_indices]

    # Fit a linear regression model: beta = (X^T X)^-1 X^T y
    beta = np.linalg.pinv(X_selected) @ y_selected

    # Predict on buyer data
    predictions = X_buy @ beta

    # Compute Mean Squared Error
    mse_error = mean_squared_error(y_buy, predictions)

    return mse_error

def identify_samples(weights, num_select=10):
    """
    Identify which data points are selected based on weights.
    """
    selected_indices = np.argsort(weights)[-num_select:]
    unsampled_indices = np.argsort(weights)[:-num_select]
    return selected_indices, unsampled_indices



