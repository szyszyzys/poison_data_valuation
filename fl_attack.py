def flattack_round(self, num_select: int = None, **kwargs):
    """
    Similar to train_federated_round or martFL,
    but we assume some sellers are adversarial and provide malicious updates.
    """
    # 1. Get gradients from each seller
    gradients, sizes, seller_ids = self.get_current_market_gradients()

    # 2. Possibly filter or cluster out outliers, or do normal FedAvg
    selected_grads, selected_sizes, selected_sellers = self.select_gradients(
        gradients, sizes, seller_ids, num_select=num_select, **kwargs
    )

    # 3. Aggregate the malicious + honest updates
    agg_gradient = self.aggregate_gradients(selected_grads, selected_sizes)

    # 4. Update global model
    self.update_global_model(agg_gradient)

    # 5. Broadcast
    self.broadcast_global_model()

    # Return info
    return {
        "aggregated_gradient": agg_gradient,
        "used_sellers": selected_sellers,
    }
