import pandas as pd
import dask.dataframe as dd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import logging

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    filename="logs/etl_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def transform_data(input_file):
    """Clean, normalize, and engineer features in customer data."""

    logging.info("Starting transform phase.")

    # Read raw data with Dask and explicitly define nullable numeric types
    ddf = dd.read_csv(
        input_file,
        dtype={
            "age": "float64",
            "monthly_spend": "float64"
        }
    )

    print("Dask partitions:", ddf.npartitions)

    # Convert to pandas after Dask ingestion
    df = ddf.compute()

    print("\nMissing values BEFORE cleaning:")
    print(df.isnull().sum())

    # Handle missing values
    df["age"] = df["age"].fillna(df["age"].median())
    df["monthly_spend"] = df["monthly_spend"].fillna(
        df["monthly_spend"].median()
    )

    # Encode categorical variable
    df["contract_encoded"] = df["contract_type"].map(
        {"Monthly": 0, "Annual": 1}
    )

    # Feature engineering
    df["spend_per_month"] = (
        df["monthly_spend"] / df["tenure_months"].clip(lower=1)
    )

    df["high_support_usage"] = (
        df["support_calls"] >= 4
    ).astype(int)

    # Normalize numerical features
    scaler = MinMaxScaler()

    columns_to_scale = [
        "age",
        "monthly_spend",
        "tenure_months",
        "support_calls"
    ]

    scaled_values = scaler.fit_transform(df[columns_to_scale])

    for index, column in enumerate(columns_to_scale):
        df[f"{column}_normalized"] = scaled_values[:, index]

    print("\nMissing values AFTER cleaning:")
    print(df.isnull().sum())

    output_file = "data/processed/customer_data_transformed.csv"
    df.to_csv(output_file, index=False)

    logging.info(
        "Transform phase completed successfully with %d records.",
        len(df)
    )

    return df


if __name__ == "__main__":
    transformed_data = transform_data(
        "data/raw/customer_data.csv"
    )

    print("\nETL TRANSFORM PHASE SUCCESSFUL")
    print("------------------------------")
    print(f"Records transformed: {len(transformed_data)}")

    print("\nEngineered and normalized features:")
    print(
        transformed_data[
            [
                "customer_id",
                "age",
                "monthly_spend",
                "contract_encoded",
                "spend_per_month",
                "high_support_usage",
                "age_normalized",
                "monthly_spend_normalized"
            ]
        ].head(10)
    )

    print(
        "\nTransformed data saved to: "
        "data/processed/customer_data_transformed.csv"
    )
