import polars as pl


def rm_null_columns(df: pl.DataFrame) -> pl.DataFrame:
    has_nulls = df.select(pl.all().is_null().any()).to_dict()
    selection = []
    for name, has_nulls in has_nulls.items():
        if not has_nulls.item():
            selection.append(name)

    return df.select(selection)


def workflow(path, write=False):

    df = pl.read_csv(
        path,
        has_header=False,
    )

    df = df.pipe(rm_null_columns)

    df.write_csv(
        path.with_stem("volumes_no_nans"),
        float_scientific=True,
        float_precision=10,
        include_header=False,
    )

    return df


def main():
    from mytorch.io.readers import read_study

    study = read_study("./config.toml")
    volumes = study.paths.volumes_collected
    times = study.paths.times_collected

    df_vol = workflow(volumes, write=True)
    df_time = workflow(times, write=True)

    has_zeros = df_time.select(col for col in df_time.select(pl.all().eq(0).any()))
    print(has_zeros)
    print(df_time.filter(pl.col("column_4").eq(0)))


if __name__ == "__main__":
    main()
