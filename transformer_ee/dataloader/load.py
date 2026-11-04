"""
Load the data
"""

import numpy as np
import torch

from transformer_ee.dataloader.noise import normalized_noise


def get_sample_indices(sample_size: int, config) -> tuple:
    """
    USED BY PANDAS.
    A function to get the indices of the sample rows from a whole Pandas DataFrame.

    sample_size:    the number of samples
    config:         the configuration dictionary
    return:         the indices of train, validation and test sets
    """

    seed = config["seed"]

    _indices = np.arange(sample_size)
    np.random.seed(seed)
    np.random.shuffle(_indices)

    test_size = config["test_size"]
    valid_size = config["valid_size"]

    # test_size and valid_size can be either int or float
    if isinstance(test_size, float):
        test_size = int(sample_size * test_size)
    if isinstance(valid_size, float):
        valid_size = int(sample_size * valid_size)

    train_indicies = _indices[: sample_size - valid_size - test_size]
    valid_indicies = _indices[
        sample_size - valid_size - test_size : sample_size - test_size
    ]
    test_indicies = _indices[sample_size - test_size :]

    print("train indicies size:\t", len(train_indicies))
    print("valid indicies size:\t", len(valid_indicies))
    print("test  indicies size:\t", len(test_indicies))

    return train_indicies, valid_indicies, test_indicies


def get_sample_sizes(sample_size: int, config) -> tuple:
    """
    USED BY POLARS.
    A function to get the sizes of the samples.

    sample_size:    the number of samples
    config:         the configuration dictionary
    return:         a tuple of three integers:
                        the number of training samples,
                        the number of validation samples,
                        the number of test samples
    """

    test_size = config["test_size"]
    valid_size = config["valid_size"]

    # test_size and valid_size can be either int or float
    if isinstance(test_size, float):
        test_size = int(sample_size * test_size)
    if isinstance(valid_size, float):
        valid_size = int(sample_size * valid_size)

    print("train indicies size:\t", sample_size - test_size - valid_size)
    print("valid indicies size:\t", valid_size)
    print("test  indicies size:\t", test_size)

    return sample_size - test_size - valid_size, valid_size, test_size


def get_train_valid_test_dataloader(config: dict):  # pylint: disable=R0914
    """
    A function to get the train, validation and test datasets
    Use the statistic of the training set to normalize the validation and test sets
    """

    dataframe_type = config.get("dataframe_type", "pandas")
    if config.get("subsample") and not config.get("split_column"):  # never ignore a requested subsample silently
        raise ValueError("subsample requires split_column (polars)")
    if dataframe_type == "pandas":
        import pandas as pd  # pylint: disable=C0415

        from transformer_ee.dataloader.pd_dataset import (  # pylint: disable=C0415
            Normalized_pandas_Dataset_with_cache,
        )

        if config.get("split_column"):
            raise ValueError("split_column is only supported with dataframe_type 'polars'")
        df = pd.read_csv(config["data_path"])
        train_idx, valid_idx, test_idx = get_sample_indices(len(df), config)
        train_set = Normalized_pandas_Dataset_with_cache(
            config, df.iloc[train_idx].reset_index(drop=True, inplace=False)
        )
        valid_set = Normalized_pandas_Dataset_with_cache(
            config,
            df.iloc[valid_idx].reset_index(drop=True, inplace=False),
            weighter=train_set.weighter,
        )
        test_set = Normalized_pandas_Dataset_with_cache(
            config,
            df.iloc[test_idx].reset_index(drop=True, inplace=False),
            weighter=train_set.weighter,
        )
    elif dataframe_type == "polars":
        from transformer_ee.dataloader.pl_dataset import (  # pylint: disable=C0415
            Normalized_Polars_Dataset_with_cache,
            get_polars_df_from_file,
        )

        df = get_polars_df_from_file(config["data_path"])
        split_column = config.get("split_column")
        if split_column:
            # Pre-assigned split (e.g. stratified per sample upstream).
            # Rows keep file order, so result.csv lines up with the file's test rows.
            import polars as pl  # pylint: disable=C0415

            labels = set(df.get_column(split_column).unique().to_list())
            if not labels <= {"train", "valid", "test"}:
                raise ValueError(f"{split_column} has labels {labels}; expected train/valid/test")
            sub = config.get("subsample")
            if sub:
                # Lets one input sample's size be reduced when several samples are trained together:
                # drops a hashed share of its train/valid rows; test rows are never touched.
                if "test" in sub["splits"]:
                    raise ValueError("subsample must not touch test rows")
                from pcnumu.prep.make_split_csv import splitmix64  # same hash as the split column  # pylint: disable=C0415

                run, srun, cyc, evt = (df[c].cast(pl.UInt64).to_numpy() for c in ("run", "subrun", "cycle", "evt"))
                key = ((run * np.uint64(1000) + srun) * np.uint64(1000) + cyc) * np.uint64(10**6) + evt
                with np.errstate(over="ignore"):
                    u = splitmix64(key ^ np.uint64(sub["seed"])).astype(np.float64) / 2.0**64
                drop = ((df["sample"].to_numpy() == sub["sample"])
                        & df[split_column].is_in(sub["splits"]).to_numpy() & (u >= sub["keep"]))
                df = df.filter(pl.Series(~drop))
                print("subsample dropped:\t", int(drop.sum()), sub)
            train_df, valid_df, test_df = (
                df.filter(pl.col(split_column) == name) for name in ("train", "valid", "test")
            )
            del df
            print("split from column:\t", split_column)
            print("train indicies size:\t", train_df.height)
            print("valid indicies size:\t", valid_df.height)
            print("test  indicies size:\t", test_df.height)
        else:
            randomdf = df.sample(fraction=1.0, seed=config["seed"], shuffle=True)
            del df
            sizes = get_sample_sizes(randomdf.height, config)
            train_df = randomdf.slice(offset=0, length=sizes[0])
            valid_df = randomdf.slice(offset=sizes[0], length=sizes[1])
            test_df = randomdf.slice(offset=sizes[0] + sizes[1], length=sizes[2])
        train_set = Normalized_Polars_Dataset_with_cache(config, train_df)
        valid_set = Normalized_Polars_Dataset_with_cache(
            config,
            valid_df,
            weighter=train_set.weighter,
        )
        test_set = Normalized_Polars_Dataset_with_cache(
            config,
            test_df,
            weighter=train_set.weighter,
        )

    else:
        raise ValueError(
            f"Unknown dataframe_type: {dataframe_type}. Supported types: ['pandas', 'polars']"
        )

    train_set.statistic()
    train_set.normalize()
    valid_set.normalize(train_set.stat)
    test_set.normalize(train_set.stat)

    batch_size_train = config["batch_size_train"]
    batch_size_valid = config["batch_size_valid"]
    batch_size_test = config["batch_size_test"]

    train_collate_fn = None
    if "noise" in config:
        train_collate_fn = normalized_noise(config, train_set.stat)

    trainloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=config.get("num_workers", 10),
        collate_fn=train_collate_fn,
    )

    validloader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=batch_size_valid,
        shuffle=False,
        num_workers=config.get("num_workers", 10),
    )

    testloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=config.get("num_workers", 10),
    )

    return trainloader, validloader, testloader, train_set.stat
