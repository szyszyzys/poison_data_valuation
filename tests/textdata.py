from common.datasets.text_data_processor import get_text_data_set


def test_get_text_data_set():
    """
    Smoke‐test for get_text_data_set:
    - loads AG_NEWS and TREC
    - checks that class_names and vocab look sane
    - pulls one batch from buyer, sellers, and test dataloaders
    """
    for name in ("AG_NEWS", "TREC"):
        print(f"\n=== Testing {name} ===")
        buyer_loader, seller_loaders, test_loader, class_names, vocab, pad_idx = get_text_data_set(
            dataset_name=name,
            batch_size=8,
            data_root=".data",
            seed=42
        )

        # basic metadata
        print(f"{name}: {len(class_names)} classes → {class_names}")
        print(f"Vocab size: {len(vocab)}, pad_idx: {pad_idx}")

        # helper to pull a batch and print shapes
        def show_loader(loader, label):
            if loader is None:
                print(f"  {label}: None")
                return
            it = iter(loader)
            batch = next(it)
            # assumes batch is a tuple (labels, texts) or dict of tensors
            if isinstance(batch, (list, tuple)):
                shapes = [t.shape for t in batch if hasattr(t, 'shape')]
                print(f"  {label} batch shapes: {shapes}")
            else:
                print(f"  {label} batch: {batch}")

        # buyer
        show_loader(buyer_loader, "buyer_loader")

        # sellers
        for client_id, sl in seller_loaders.items():
            show_loader(sl, f"seller[{client_id}]")

        # test
        show_loader(test_loader, "test_loader")

if __name__ == "__main__":
    test_get_text_data_set()