import polars as pl

from mytorch.io.readers import read_study

# Read the study
study = read_study("./config.toml")
params = study.paths.parameters

# scan the data and append each parameter as a row to the dataframe
samples = [i for i in range(1023)]

queries = []
for sample in samples:
    queries.append(
        pl.scan_csv(str(params).replace("*", str(sample)), has_header=False).select(
            pl.col("column_1").alias("value"),
            pl.int_range(12).alias("feature"),
            pl.lit(sample).alias("sample"),
        )
    )

# Concatenate the queries
df = (
    pl.concat(queries)
    .collect()
    .pivot("feature", index="sample", values="value")
    .drop("sample")
)
df.write_csv(
    study.paths.processed / "parameters.csv",
    float_scientific=True,
    include_header=False,
)
print(df.describe())
