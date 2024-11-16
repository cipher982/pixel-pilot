import torch


def get_device():
    """Get the best available device for PyTorch."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    print(get_device())
