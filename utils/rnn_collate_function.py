import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    use collate_fn to pad sequences to the same length within a batch
    """
    # batch is a list of (sequence, target) pairs
    sequences, targets = zip(*batch)

    # use pad_sequence to pad sequences to the same length
    padded_sequences = pad_sequence(
        sequences, batch_first=True, padding_value=0.0, padding_side="left"
    )
    padded_targets = pad_sequence(
        targets, batch_first=True, padding_value=0.0, padding_side="left"
    )
    # lengths = torch.tensor(
    #     [(len(seq) - 1) for seq in sequences]
    """
    we use time <= (t-1)'s features to predict time = t's target
    """
    lengths = torch.tensor(
        [(len(seq)-1) for seq in sequences]
        # [seq.shape[0] for seq in sequences]
    )  # get the original length of each sequence
    return padded_sequences, padded_targets, lengths
