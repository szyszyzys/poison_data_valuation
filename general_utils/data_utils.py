from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, Dataset


# Optional: Create a simple Dataset class (good practice)
class TextDataset(Dataset):
    def __init__(self, data: List[Tuple[int, List[int]]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # Returns (label, list_of_token_ids)


# Assume 'vocab' and 'pad_idx' are available from your data loading step
# vocab = loaded_data['vocabulary']
# pad_idx = loaded_data['padding_idx']

def collate_batch(
        batch: List[Tuple[int, List[int]]],  # Batch is a list of (label, token_ids)
        padding_value: int  # Pass the padding index from your vocab
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # Returns labels, padded_sequences, lengths
    """
    Collates a batch of text data, padding sequences.

    Args:
        batch: A list of tuples, where each tuple is (label, list_of_token_ids).
        padding_value: The integer index used for padding.

    Returns:
        A tuple containing:
            - labels_tensor (Tensor): Tensor of labels for the batch.
            - padded_sequences (Tensor): Tensor of token ID sequences, padded to
                                        the length of the longest sequence in the batch.
                                        Shape: (batch_size, max_seq_length)
            - lengths_tensor (Tensor): Tensor of original sequence lengths for each item.
                                        Shape: (batch_size,) Useful for PackedSequence.
    """
    labels_list, sequences_list, lengths_list = [], [], []
    for (label, token_ids) in batch:
        labels_list.append(label)
        # Convert list of token IDs to a Tensor for this sequence
        seq_tensor = torch.tensor(token_ids, dtype=torch.int64)
        sequences_list.append(seq_tensor)
        lengths_list.append(len(token_ids))  # Store original length

    labels_tensor = torch.tensor(labels_list, dtype=torch.int64)
    lengths_tensor = torch.tensor(lengths_list, dtype=torch.int64)

    # Pad sequences
    # pad_sequence expects a list of Tensors and pads them
    # batch_first=True makes the output shape (batch_size, max_seq_length)
    padded_sequences = pad_sequence(
        sequences_list, batch_first=True, padding_value=padding_value
    )

    return labels_tensor, padded_sequences, lengths_tensor


def list_to_tensor_dataset(data: list[tuple[torch.Tensor, int]]) -> TensorDataset:
    # Separate images and labels from the list of tuples.
    images = torch.stack([img for img, label in data])
    labels = torch.tensor([label for img, label in data])
    return TensorDataset(images, labels)
