import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


def create_train_test_group(
    data: pd.DataFrame,
    id_column: str = "id",
    sample_frac: float = 1.0,
    test_size: float = 0.1,
    valid_size: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Adds a 'group' column to the DataFrame, assigning 'train' or 'test' based on a train/test split of unique IDs.
    This function is optimized for speed using numpy and vectorized operations.

    Args:
        data (pd.DataFrame): The input DataFrame.
        id_column (str): The name of the column containing unique IDs. Defaults to "id".
        test_size (float): The proportion of unique IDs to include in the test set. Defaults to 0.1.
        valid_size (float): Proportion of unique IDs to include in the validation set. Defaults to 0.1.
        random_state (int): The random state for the train/test split. Defaults to 42.

    Returns:
        pd.DataFrame: The DataFrame with the added 'group' column.
    """
    # unique_ids = data[id_column].unique()
    # # select n% of the unique_ids
    # if sample_frac < 1.0:
    #     np.random.seed(random_state)
    #     unique_ids = np.random.choice(
    #         unique_ids,
    #         size=int(len(unique_ids) * sample_frac),
    #         replace=False,
    #     )
    #     unique_id_set = set(unique_ids)
    #     data = data[data[id_column].isin(unique_id_set)].copy()
    # # Create train and test sets

    # train_ids, test_ids = train_test_split(
    #     unique_ids, test_size=test_size, random_state=random_state
    # )

    # # Convert test_ids to a set for faster membership checking
    # test_ids_set = set(test_ids)

    # # Use numpy's vectorized apply for faster group assignment
    # data["group"] = np.where(data[id_column].isin(test_ids_set), "test", "train")

    # return data
    unique_ids = data[id_column].unique()

    # Sample a fraction of unique IDs if requested
    if sample_frac < 1.0:
        np.random.seed(random_state)
        unique_ids = np.random.choice(
            unique_ids,
            size=int(len(unique_ids) * sample_frac),
            replace=False,
        )
        unique_id_set = set(unique_ids)
        data = data[data[id_column].isin(unique_id_set)].copy()

    # Split unique IDs into train, test, and validation sets
    total_holdout = test_size + valid_size
    train_ids, temp_ids = train_test_split(
        unique_ids, test_size=total_holdout, random_state=random_state
    )
    test_ids, valid_ids = train_test_split(
        temp_ids,
        test_size=valid_size / total_holdout,
        random_state=random_state,
    )

    # Assign groups using vectorized operations
    test_ids_set = set(test_ids)
    valid_ids_set = set(valid_ids)
    data["group"] = np.where(
        data[id_column].isin(test_ids_set),
        "test",
        np.where(data[id_column].isin(valid_ids_set), "valid", "train"),
    )

    return data