# # !/usr/bin/env python3
# """
# text_models.py
#
# This file includes:
#
# 1. Two model definitions for text classification:
#    - CNN_MLP: An embedding layer followed by a convolution, pooling and MLP head.
#    - TEXTCNN: A multi-channel CNN (using multiple kernel sizes) as in Kim (2014).
#
# 2. A data loader function that works for AG_NEWS and TREC datasets using torchtext.
#    It builds the vocabulary, tokenizes, pads sequences, and returns train and test DataLoaders.
#
# 3. Utility functions to save and load models (to support saving the model each round in FL).
#
# Usage Example:
# --------------
#     from text_models import (
#          CNN_MLP, TEXTCNN, get_dataloaders, save_model, load_model
#     )
#     # Choose dataset: "AG_NEWS" (4 classes) or "TREC" (6 classes)
#     train_loader, test_loader, vocab = get_dataloaders(dataset_name="AG_NEWS", batch_size=32, max_seq_len=100)
#     num_classes = 4 if "AG_NEWS" in "AG_NEWS".upper() else 6
#     embed_dim = 100
#
#     # Instantiate one of the models (e.g., TEXTCNN)
#     model = TEXTCNN(vocab_size=len(vocab), embed_dim=embed_dim, num_classes=num_classes)
#     model.to(device)
#
#     # ... train the model ...
#
#     # Save the model after the round:
#     save_model(model, "saved_models/model_round1.pt")
#     # Later, load it back with:
#     model = load_model(model, "saved_models/model_round1.pt", device=device)
#
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# # from torchtext.data.utils import get_tokenizer
# # from torchtext.datasets import AG_NEWS, TREC
# # from torchtext.vocab import build_vocab_from_iterator
# #
#
# # ---------------------------
# # Model Definitions
# # ---------------------------
#
# class CNN_MLP(nn.Module):
#     """
#     CNN+MLP model for text classification.
#     Consists of an embedding layer, a 1D convolution, ReLU, adaptive max-pooling,
#     and a two-layer MLP.
#     """
#
#     def __init__(self, vocab_size, embed_dim, num_classes,
#                  kernel_size=3, num_filters=100, hidden_dim=128):
#         super(CNN_MLP, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         # The conv is applied on the embedded sequence (channels=embed_dim)
#         self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters,
#                               kernel_size=kernel_size, padding=1)
#         self.relu = nn.ReLU()
#         self.pool = nn.AdaptiveMaxPool1d(1)
#         self.fc1 = nn.Linear(num_filters, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, num_classes)
#
#     def forward(self, x):
#         # x: (batch_size, seq_length)
#         embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
#         embedded = embedded.transpose(1, 2)  # (batch, embed_dim, seq_len)
#         conv_out = self.conv(embedded)  # (batch, num_filters, seq_len)
#         conv_out = self.relu(conv_out)
#         pooled = self.pool(conv_out).squeeze(-1)  # (batch, num_filters)
#         hidden = self.relu(self.fc1(pooled))
#         out = self.fc2(hidden)
#         return out
#
#
# # class TEXTCNN(nn.Module):
# #     """
# #     TEXTCNN as in Kim (2014) for sentence classification.
# #     Uses multiple convolution filters with different kernel sizes followed by max pooling.
# #     """
# #
# #     def __init__(self, vocab_size, embed_dim, num_classes,
# #                  kernel_sizes=[3, 4, 5], num_filters=100, dropout=0.5):
# #         super(TEXTCNN, self).__init__()
# #         self.embedding = nn.Embedding(vocab_size, embed_dim)
# #         # Create a convolution for each kernel size
# #         self.convs = nn.ModuleList([
# #             nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
# #             for k in kernel_sizes
# #         ])
# #         self.dropout = nn.Dropout(dropout)
# #         self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
# #
# #     def forward(self, x):
# #         # x: (batch_size, seq_length)
# #         embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
# #         embedded = embedded.transpose(1, 2)  # (batch, embed_dim, seq_len)
# #         convs = [F.relu(conv(embedded)) for conv in self.convs]  # list of (batch, num_filters, L_out)
# #         pools = [F.max_pool1d(conv_out, kernel_size=conv_out.shape[2]).squeeze(2)
# #                  for conv_out in convs]  # each: (batch, num_filters)
# #         cat = torch.cat(pools, dim=1)  # (batch, num_filters * len(kernel_sizes))
# #         drop = self.dropout(cat)
# #         out = self.fc(drop)
# #         return out
# #
# #
# # # ---------------------------
# # # Data Loader Functions
# # # ---------------------------
# #
# # tokenizer = get_tokenizer("basic_english")
# #
# #
# # def yield_tokens(data_iter):
# #     for label, text in data_iter:
# #         yield tokenizer(text)
# #
# #
# # def build_vocab(dataset_iter, min_freq=1):
# #     vocab = build_vocab_from_iterator(yield_tokens(dataset_iter), specials=["<unk>"], min_freq=min_freq)
# #     vocab.set_default_index(vocab["<unk>"])
# #     return vocab
#
#
# # def collate_batch(batch, vocab, max_seq_len=100):
# #     """
# #     Collate function for text batches.
# #     Each batch element is a tuple (label, text).
# #     We tokenize, convert tokens to indices, cut off/pad to max_seq_len.
# #     """
# #     labels = []
# #     text_list = []
# #     for (label, text) in batch:
# #         # For AG_NEWS, labels are 1-indexed (1-4). For TREC, labels are strings.
# #         if isinstance(label, int):
# #             labels.append(label - 1)
# #         else:
# #             # For TREC, we map string labels to int. Example mapping:
# #             mapping = {"DESC": 0, "ENTY": 1, "ABBR": 2, "HUM": 3, "LOC": 4, "NUM": 5}
# #             labels.append(mapping[label])
# #         tokens = tokenizer(text)
# #         token_ids = vocab(tokens)
# #         token_ids = token_ids[:max_seq_len]
# #         text_list.append(torch.tensor(token_ids, dtype=torch.long))
# #     # Pad sequences (batch_first=True) using 0 as padding index.
# #     text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=0)
# #     labels = torch.tensor(labels, dtype=torch.long)
# #     return text_list, labels
#
#
# def get_dataloaders(dataset_name="AG_NEWS", batch_size=32, max_seq_len=100):
#     """
#     Returns (train_loader, test_loader, vocab) for the specified dataset.
#
#     Supported dataset_name:
#       - "AG_NEWS": 4-class news classification.
#       - "TREC": 6-class question classification.
#     """
#     dataset_name = dataset_name.upper()
#     if dataset_name == "AG_NEWS":
#         train_iter, test_iter = AG_NEWS()
#     elif dataset_name == "TREC":
#         # TREC returns (label, text); available in recent torchtext versions.
#         train_iter, test_iter = TREC(split=('train', 'test'))
#     else:
#         raise ValueError("Unsupported dataset. Choose AG_NEWS or TREC.")
#
#     # Convert train_iter to a list to build vocabulary.
#     train_list = list(train_iter)
#     vocab = build_vocab(train_list, min_freq=2)
#
#     def process_data(data_iter):
#         texts = []
#         labels = []
#         for (label, text) in data_iter:
#             if dataset_name == "AG_NEWS":
#                 labels.append(label - 1)
#             else:
#                 mapping = {"DESC": 0, "ENTY": 1, "ABBR": 2, "HUM": 3, "LOC": 4, "NUM": 5}
#                 labels.append(mapping[label])
#             tokens = tokenizer(text)
#             token_ids = vocab(tokens)
#             token_ids = token_ids[:max_seq_len]
#             texts.append(torch.tensor(token_ids, dtype=torch.long))
#         texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
#         labels = torch.tensor(labels, dtype=torch.long)
#         return TensorDataset(texts, labels)
#
#     train_dataset = process_data(train_list)
#     test_list = list(test_iter)
#     test_dataset = process_data(test_list)
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                               collate_fn=lambda b: collate_batch(b, vocab, max_seq_len))
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
#                              collate_fn=lambda b: collate_batch(b, vocab, max_seq_len))
#     return train_loader, test_loader, vocab
#
#
# # ---------------------------
# # Example Main (for testing)
# # ---------------------------
#
# if __name__ == "__main__":
#     # Parameters
#     dataset_name = "AG_NEWS"  # or "TREC"
#     batch_size = 32
#     max_seq_len = 100
#     embed_dim = 100
#     model_type = "TEXTCNN"  # or "CNN_MLP"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # Get DataLoaders and Vocabulary
#     train_loader, test_loader, vocab = get_dataloaders(dataset_name=dataset_name,
#                                                        batch_size=batch_size,
#                                                        max_seq_len=max_seq_len)
#     print("Vocabulary size:", len(vocab))
#
#     # Determine number of classes
#     num_classes = 4 if dataset_name == "AG_NEWS" else 6
#
#     # Instantiate the model
#     if model_type.upper() == "CNN_MLP":
#         model = CNN_MLP(vocab_size=len(vocab), embed_dim=embed_dim, num_classes=num_classes)
#     else:
#         model = TEXTCNN(vocab_size=len(vocab), embed_dim=embed_dim, num_classes=num_classes)
#
#     model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()
#
#     # Train for one epoch (for demonstration)
#     model.train()
#     for batch_idx, (texts, labels) in enumerate(train_loader):
#         texts = texts.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(texts)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 10 == 0:
#             print(f"Batch {batch_idx} Loss: {loss.item():.4f}")
#
#     # Save the model for this round
#     save_model(model, "saved_models/model_round1.pt")
#
#     # Optionally, load the model back
#     model = load_model(model, "saved_models/model_round1.pt", device=device)
