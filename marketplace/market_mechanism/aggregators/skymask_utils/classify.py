import numpy as np
from sklearn import mixture
from sklearn.decomposition import PCA


def GMM2(mask_list):
    if not mask_list or len(mask_list) < 2:
        print("Warning: GMM requires at least 2 samples. Returning default labels.")
        return [0] * len(mask_list)  # Mark all as outliers if not enough data

    # Ensure input is a numpy array for sklearn
    try:
        mask_array = np.array(mask_list)
        if mask_array.ndim == 1:  # If PCA gets 1D data (e.g., only 1 sample after filtering)
            print("Warning: PCA/GMM input is 1D. Adjusting.")
            # Handle 1D case: maybe skip PCA or return default
            # For now, let's assume GMM can handle 1D directly if n_components=1 or handle error
            # Let's return default for simplicity in this edge case
            return [0] * len(mask_list)

        # Proceed with PCA only if input is suitable
        if mask_array.shape[1] < 2:  # Check if feature dimension is less than n_components
            print("Warning: Feature dimension < 2. Skipping PCA.")
            newX = mask_array
            n_components_gmm = min(2, mask_array.shape[0])  # Adjust GMM components based on samples
        else:
            pca = PCA(n_components=2)
            newX = pca.fit_transform(mask_array)
            n_components_gmm = 2

        # Ensure n_components <= n_samples for GMM
        n_components_gmm = min(n_components_gmm, newX.shape[0])
        if n_components_gmm < 1:
            print("Warning: Not enough samples for GMM after PCA. Returning defaults.")
            return [0] * len(mask_list)

        gmm = mixture.GaussianMixture(n_components=n_components_gmm, random_state=0).fit(newX)  # Added random_state
        y_pred = gmm.predict(newX)

        # --- Post-processing logic ---
        # If GMM used only 1 component, all predictions will be 0. Interpret appropriately.
        if n_components_gmm == 1:
            print("GMM ran with 1 component. Interpretation might need adjustment.")
            # Decide if 1 component means all inliers or requires different handling. Assume inliers for now.
            return [1] * len(mask_list)

        # Recalculate benign/mali based on actual predictions
        benign = np.sum(y_pred == 1)
        mali = np.sum(y_pred == 0)

        # Check if the majority class needs flipping - THIS LOGIC IS SUSPICIOUS
        # The original code flips if the *last* element is 0. This seems arbitrary.
        # A more common approach is to assume the larger cluster is benign (1).
        if mali > benign:  # If cluster 0 is larger, flip labels so 1 represents the larger cluster
            print("Flipping GMM labels: Assuming larger cluster is benign (1).")
            y_pred = 1 - y_pred  # Flip 0s to 1s and 1s to 0s

        return y_pred.tolist()  # Return as list

    except Exception as e:
        print(f"Error during GMM/PCA: {e}")
        print(f"Input mask_list shapes: {[m.shape for m in mask_list]}")
        # Handle error: maybe return all outliers
        return [0] * len(mask_list)
