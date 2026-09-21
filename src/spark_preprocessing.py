from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def build_spark():
    return (SparkSession.builder
            .appName("lhind-preprocessing")
            .master("local[*]")
            .getOrCreate())


def load_raw(spark):
    schema = "Date DATE, store INT, product INT, number_sold INT"
    train = spark.read.csv(str(RAW/"train.csv"), header=True, schema=schema)
    test = spark.read.csv(str(RAW/"test.csv"), header=True, schema=schema)
    return train.unionByName(test)


def audit(df):
    """Same checks as EDA Finding 1 (completeness, nulls, duplicates), just computed in Spark."""
    total_rows = df.count()
    n_days = df.select("Date").distinct().count()
    n_pairs = df.select("store", "product").distinct().count()
    span = df.select(F.datediff(F.max("Date"), F.min("Date"))).first()[0] + 1
    null_values = sum(df.filter(F.col(c).isNull()).count() for c in df.columns)
    duplicate_keys = total_rows - df.dropDuplicates(["Date", "store", "product"]).count()
    non_positive_sales = df.filter(F.col("number_sold") <= 0).count()
    return {
        "rows": total_rows,
        "unique_dates": n_days,
        "calendar_days_in_range": span,
        "store_product_pairs": n_pairs,
        "rows_if_complete": n_days * n_pairs,
        "null_values": null_values,
        "duplicate_keys": duplicate_keys,
        "negative_or_zero_sales": non_positive_sales,
    }


def add_features(df):
    return (df
            .withColumn("year", F.year("Date"))
            .withColumn("month", F.month("Date"))
            .withColumn("day_of_year", F.dayofyear("Date"))
            .withColumn("day_of_week", (F.dayofweek("Date") + 5) % 7)
            .withColumn("pair", F.col("store") * 10 + F.col("product")))


def pair_descriptive_stats(df):
    """Core per-pair stats from EDA section 3 (mean/spread/growth), computed in Spark."""
    stats = (df.groupBy("store", "product")
             .agg(F.mean("number_sold").alias("mean"),
                  F.stddev("number_sold").alias("std"),
                  F.min("number_sold").alias("min"),
                  F.expr("percentile_approx(number_sold, 0.25)").alias("q25"),
                  F.expr("percentile_approx(number_sold, 0.5)").alias("median"),
                  F.expr("percentile_approx(number_sold, 0.75)").alias("q75"),
                  F.max("number_sold").alias("max"))
             .withColumn("variation_coefficient_pct", F.col("std") / F.col("mean") * 100))

    yearly = (df.filter(F.col("year").isin(2010, 2018))
              .groupBy("store", "product")
              .pivot("year", [2010, 2018])
              .agg(F.mean("number_sold"))
              .withColumn("growth_2010_2018_pct", (F.col("2018") / F.col("2010") - 1) * 100)
              .select("store", "product", "growth_2010_2018_pct"))

    return stats.join(yearly, ["store", "product"]).orderBy("store", "product")


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    try:
        raw = load_raw(spark)

        print("data quality audit:")
        for key, value in audit(raw).items():
            print(f"  {key}: {value}")

        features = add_features(raw)
        stats = pair_descriptive_stats(features.filter(F.col("year") <= 2018))

        PROCESSED.mkdir(parents=True, exist_ok=True)
        features.toPandas().to_csv(PROCESSED / "spark_features.csv", index=False)
        stats.toPandas().round(2).to_csv(PROCESSED / "spark_series_descriptive_stats.csv", index=False)

        print(f"wrote {PROCESSED / 'spark_features.csv'}")
        print(f"wrote {PROCESSED / 'spark_series_descriptive_stats.csv'}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
