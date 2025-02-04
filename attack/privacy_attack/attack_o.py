import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

###############################################################################
#                               Attack Code                                   #
###############################################################################

def compute_fisher_information(selected_data: torch.Tensor, lambda_reg: float = 1e-3):
    """
    Computes the Fisher information matrix:
        I = sum_i (x_i x_i^T) + lambda_reg * I_d
    and returns I_inv (the inverse).

    selected_data: (n_selected, d)
    lambda_reg   : small float for regularization
    Returns: I_inv of shape (d, d)
    """
    device = selected_data.device
    d = selected_data.shape[1]

    # Sum up outer products
    I = torch.zeros(d, d, device=device)
    for x in selected_data:
        x = x.view(1, d)
        I += x.t() @ x
    # Regularization
    I += lambda_reg * torch.eye(d, device=device)

    return torch.inverse(I)

def score_function(x_query: torch.Tensor, I_inv: torch.Tensor, candidate_data: torch.Tensor):
    """
    Given a single query vector x_query (1, d), compute the "score" for each candidate x_j:
       score_j = (x_query^T I_inv x_j)^2

    x_query      : (1, d)
    I_inv        : (d, d)
    candidate_data: (N, d)
    Returns: (N,) vector of scores
    """
    # (1,d) @ (d,d) = (1,d)
    Q = x_query @ I_inv
    # (1,d) @ (d,N) = (1,N) => square => shape: (1,N)
    scores = (Q @ candidate_data.t())**2
    # Return as (N,)
    return scores.flatten()

def ranking_loss(pred_scores: torch.Tensor, selected_mask: torch.Tensor, margin: float = 0.1):
    """
    Pairwise hinge-style ranking loss:
       For each selected candidate, its score should exceed that of unselected candidates by 'margin'.

    pred_scores  : (N,) predicted scores
    selected_mask: (N,) binary (1=selected, 0=not selected)
    margin       : margin for ranking
    Returns: scalar loss
    """
    selected_scores = pred_scores[selected_mask == 1]
    unselected_scores = pred_scores[selected_mask == 0]
    if selected_scores.numel() == 0 or unselected_scores.numel() == 0:
        return torch.tensor(0.0, device=pred_scores.device)

    # We want: selected_scores >= unselected_scores + margin
    # diff shape => (num_sel, num_unsel)
    diff = selected_scores.unsqueeze(1) - unselected_scores.unsqueeze(0)
    loss_mat = F.relu(margin - diff)
    return loss_mat.mean()

def reconstruct_query(
    full_seller_data: torch.Tensor,
    selected_indices: torch.Tensor,
    scenario: str = 'score_known',
    observed_scores: torch.Tensor = None,
    selected_candidate_indices: torch.Tensor = None,
    lambda_reg: float = 1e-3,
    lr: float = 0.1,
    num_iters: int = 1000,
    reg_weight: float = 1e-3,
    margin: float = 0.1,
    num_restarts: int = 1,
    verbose: bool = False
):
    """
    High-level function to reconstruct an "aggregate" query vector x_query
    given a scenario: 'score_known' or 'selection_only'.

    Args:
        full_seller_data           : (N, d) All data from the seller (both selected & unselected).
        selected_indices           : Indices (within full_seller_data) that the buyer used to build Fisher info.
        scenario                   : Either 'score_known' or 'selection_only'.
        observed_scores            : Required if scenario='score_known'. Shape (N,) with a score per candidate data.
        selected_candidate_indices : Required if scenario='selection_only'. Indices or boolean mask for final chosen subset.
        lambda_reg                 : Regularization for Fisher information matrix.
        lr                         : Learning rate for Adam.
        num_iters                  : Number of optimization iterations.
        reg_weight                 : Weight for L2 regularization on x_query.
        margin                     : Margin used in ranking loss for 'selection_only' scenario.
        num_restarts               : Number of random restarts. We pick the best final solution.
        verbose                    : If True, print intermediate logs.

    Returns:
        best_x_query : (1, d) The reconstructed query vector that best explains the
                       observed pattern (scores or selection).
        best_history : A list of the final run's training loss values (for analysis).
    """
    device = full_seller_data.device
    d = full_seller_data.shape[1]

    # 1. Compute Fisher inverse from the selected data
    selected_data = full_seller_data[selected_indices]
    I_inv = compute_fisher_information(selected_data, lambda_reg=lambda_reg)

    # 2. Identify the "candidate" set: all points in full_seller_data
    #    (In some protocols, you'd differentiate between the "already selected" set
    #    vs. "candidate" set. Here we assume the entire full_seller_data is relevant
    #    for the reported scores or final selection.)
    candidate_data = full_seller_data

    # 3. Attack function for a single run
    def single_run():
        x_query = torch.randn(1, d, device=device, requires_grad=True)
        optimizer = optim.Adam([x_query], lr=lr)
        loss_history = []

        for it in range(num_iters):
            optimizer.zero_grad()
            pred_scores = score_function(x_query, I_inv, candidate_data)

            # L2 prior on x_query to keep it "reasonable"
            reg_loss = reg_weight * (x_query**2).mean()

            if scenario == 'score_known':
                # MSE with the observed scores
                loss_main = F.mse_loss(pred_scores, observed_scores)
            elif scenario == 'selection_only':
                # Ranking loss with selected vs unselected
                # Convert selected_candidate_indices to a mask if not already
                if selected_candidate_indices.dtype == torch.bool:
                    sel_mask = selected_candidate_indices
                else:
                    # We assume it's a list of indices => build a mask
                    sel_mask = torch.zeros(len(candidate_data), dtype=torch.float, device=device)
                    sel_mask[selected_candidate_indices] = 1.0
                loss_main = ranking_loss(pred_scores, sel_mask, margin=margin)
            else:
                raise ValueError("Unknown scenario. Use 'score_known' or 'selection_only'.")

            total_loss = loss_main + reg_loss
            total_loss.backward()
            optimizer.step()
            loss_history.append(total_loss.item())

            if verbose and (it+1) % 100 == 0:
                print(f"Iter {it+1}/{num_iters}, Loss={total_loss.item():.6f}")

        return x_query.detach(), loss_history

    # 4. Multi-Restart
    best_x_query = None
    best_loss = float('inf')
    best_history = None
    for r in range(num_restarts):
        x_query_hat, hist = single_run()
        final_loss = hist[-1]
        if verbose:
            print(f"Restart {r+1}/{num_restarts}, Final Loss: {final_loss:.6f}")
        if final_loss < best_loss:
            best_loss = final_loss
            best_x_query = x_query_hat
            best_history = hist

    return best_x_query, best_history


###############################################################################
#                          Example Usage (Demo)                                #
###############################################################################
if __name__ == "__main__":
    # For reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1) Create a synthetic dataset: let's say we have 100 total data points, feature dim = 5
    N = 100
    d = 5
    full_seller_data = torch.randn(N, d)   # shape (100, 5)

    # 2) Suppose the buyer *already selected* 20 of these points (randomly chosen here)
    selected_indices = torch.randperm(N)[:20]  # shape (20,)

    # 3) We compute the "true" query vector (unknown to the seller).
    x_query_true = torch.randn(1, d)

    # 4) We'll create some "observed" transparency data for demonstration:
    #    (A) Score-Known scenario: we pretend the buyer reported the score for each data point
    #        score_j = (x_query_true^T I_inv x_j)^2
    #    (B) Selection-Only scenario: the buyer eventually picks top_k based on those scores.

    # (A) Score-Known scenario
    I_inv_demo = compute_fisher_information(full_seller_data[selected_indices], lambda_reg=1e-3)
    with torch.no_grad():
        observed_scores_demo = score_function(x_query_true, I_inv_demo, full_seller_data)

    # (B) Selection-Only scenario: pick the top 10 as "selected"
    top_k = 10
    _, sorted_idx = torch.sort(observed_scores_demo, descending=True)
    selected_candidate_indices_demo = sorted_idx[:top_k]

    # ================================
    # Score-Known Attack
    # ================================
    print("\n=== Score-Known Attack ===")
    x_query_hat_known, history_known = reconstruct_query(
        full_seller_data=full_seller_data,
        selected_indices=selected_indices,
        scenario='score_known',
        observed_scores=observed_scores_demo,
        # The next arguments are optional, but you can tweak them
        lambda_reg=1e-3, lr=0.1, num_iters=500, reg_weight=1e-3,
        num_restarts=3, verbose=True
    )

    # ================================
    # Selection-Only Attack
    # ================================
    print("\n=== Selection-Only Attack ===")
    x_query_hat_selection, history_selection = reconstruct_query(
        full_seller_data=full_seller_data,
        selected_indices=selected_indices,
        scenario='selection_only',
        selected_candidate_indices=selected_candidate_indices_demo,  # top 10
        lambda_reg=1e-3, lr=0.1, num_iters=500, reg_weight=1e-3,
        margin=0.1,
        num_restarts=3, verbose=True
    )

    # ================================
    # Compare
    # ================================
    print("\n[True query vector]:")
    print(x_query_true)
    print("[Reconstructed query - Score-Known]:")
    print(x_query_hat_known)
    print("[Reconstructed query - Selection-Only]:")
    print(x_query_hat_selection)
