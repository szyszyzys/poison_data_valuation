import torchtext
import logging # Assuming you have logging configured

# --- Your existing tokenizer and yield_tokens_for_vocab setup (simplified) ---
from torchtext.data.utils import get_tokenizer
import torch
# Dummy hf_iterator for this test
def dummy_hf_iterator_text_only(limit=None):
    sample_texts = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document again?",
        "apple banana apple orange apple", # for min_freq test
        "banana orange grape",
        "unique_word_1",
        "unique_word_2",
    ]
    if limit:
        sample_texts = sample_texts[:limit]
    for i, text in enumerate(sample_texts):
        # Simulate more data for a larger test
        # for _ in range(1000 if i < 2 else 1): # Make first few docs appear often
        yield text

def yield_tokens_for_vocab_test(text_iterator_func, tokenizer_func):
    count = 0
    for text_sample in text_iterator_func():
        yield tokenizer_func(text_sample)
        count += 1
    logging.info(f"yield_tokens_for_vocab_test: Yielded {count} tokenized samples.")

# --- Main Version Check and Test ---
def check_torchtext_version_and_min_freq():
    print(f"PyTorch version: {torch.__version__}")
    print(f"TorchText version: {torchtext.__version__}")

    version_str = torchtext.__version__
    # Simple version parsing (for more complex needs, use packaging.version)
    major, minor, *_ = map(int, version_str.split('.')[:2])

    print("\n--- Testing build_vocab_from_iterator with min_freq ---")
    tokenizer = get_tokenizer('basic_english')
    unk_token, pad_token = "<unk>", "<pad>"

    # Test 1: min_freq = 1
    print("\nBuilding vocab with min_freq = 1:")
    vocab1 = torchtext.vocab.build_vocab_from_iterator(
        yield_tokens_for_vocab_test(lambda: dummy_hf_iterator_text_only(), tokenizer),
        min_freq=1,
        specials=[unk_token, pad_token]
    )
    print(f"Vocab size (min_freq=1): {len(vocab1)}")
    print(f"Vocab1 stoi (sample): {dict(list(vocab1.get_stoi().items())[:15])}") # Show some tokens

    # Test 2: min_freq = 2
    print("\nBuilding vocab with min_freq = 2:")
    vocab2 = torchtext.vocab.build_vocab_from_iterator(
        yield_tokens_for_vocab_test(lambda: dummy_hf_iterator_text_only(), tokenizer),
        min_freq=2,
        specials=[unk_token, pad_token]
    )
    print(f"Vocab size (min_freq=2): {len(vocab2)}")
    print(f"Vocab2 stoi (sample): {dict(list(vocab2.get_stoi().items())[:15])}")
    # Check if 'unique_word_1' (freq 1) is missing
    if "unique_word_1" not in vocab2.get_stoi():
        print("SUCCESS: 'unique_word_1' (freq 1) correctly excluded with min_freq=2.")
    else:
        print("ERROR: 'unique_word_1' (freq 1) present with min_freq=2. This is unexpected.")


    # Test 3: min_freq = 3
    print("\nBuilding vocab with min_freq = 3:")
    vocab3 = torchtext.vocab.build_vocab_from_iterator(
        yield_tokens_for_vocab_test(lambda: dummy_hf_iterator_text_only(), tokenizer),
        min_freq=3,
        specials=[unk_token, pad_token]
    )
    print(f"Vocab size (min_freq=3): {len(vocab3)}")
    print(f"Vocab3 stoi (sample): {dict(list(vocab3.get_stoi().items())[:15])}")
    if "apple" in vocab3.get_stoi() and "banana" not in vocab3.get_stoi():
         print("SUCCESS: 'apple' (freq 3) included, 'banana' (freq 2) correctly excluded with min_freq=3.")
    else:
        print("ERROR: 'apple' or 'banana' handling incorrect for min_freq=3.")


    print("\n--- Advice based on version ---")
    if major == 0 and minor < 6:
        print(f"Your torchtext version ({version_str}) is quite old (pre-0.6.0).")
        print("`build_vocab_from_iterator` might have different behavior or limitations.")
        print("Consider upgrading torchtext if possible. If not, you might need to use the older `torchtext.data.Field` API for vocabulary building, which handles `min_freq` differently.")
        print("The 'stuck' issue is still likely due to memory/CPU for a large number of unique tokens before `min_freq` is applied.")
    elif (major == 0 and minor >= 6) or major > 0 : # Covers 0.6.0 up to current (e.g., 0.15.x, 0.16.x etc.)
        print(f"Your torchtext version ({version_str}) should support `min_freq` correctly in `build_vocab_from_iterator`.")
        print("If the minimal test above shows `min_freq` working (i.e., vocab size decreases appropriately), then the 'stuck' issue in your main code is almost certainly due to:")
        print("1. The sheer volume of unique tokens from 120,000 documents when `min_freq` is low (e.g., 1 or 2 in your main script's call).")
        print("   `build_vocab_from_iterator` first counts ALL unique tokens, then filters. This initial counting can be the bottleneck.")
        print("2. Memory exhaustion during this initial counting phase, leading to system slowdown (swapping).")
        print("\nTo resolve in your main script:")
        print("A. **PRIMARY FIX: Significantly increase `min_freq` in your `get_text_data_set` function call.**")
        print("   Start with `min_freq=5`, then try `min_freq=10` or even `min_freq=20` for AG_NEWS if it's still slow.")
        print("   This will drastically reduce the number of unique tokens the function has to manage internally *before and after* filtering.")
        print("B. Monitor system RAM and CPU usage while your main script is in the 'Building vocabulary...' phase to check for resource exhaustion.")
        print("C. Ensure you are actually *passing* the `min_freq` argument correctly to `get_text_data_set` and that it's not being overridden or defaulted to 1 somewhere.")
    else:
        print(f"Could not definitively categorize torchtext version {version_str} for specific advice, but the general advice about `min_freq` and resource usage applies.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # For the test script's logging
    check_torchtext_version_and_min_freq()