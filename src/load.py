import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
import logging

Path("data").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    filename="logs/etl_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(input_file):
    """Load transformed customer data into SQLite using SQLAlchemy."""

    logging.info("Starting load phase.")

    # Read transformed data
    df = pd.read_csv(input_file)

    # Create SQLite database connection
    engine = create_engine("sqlite:///data/customer_database.db")

    # Load data into relational database
    df.to_sql(
        "customer_data",
        con=engine,
        if_exists="replace",
        index=False
    )

    logging.info(
        "Loaded %d records into SQLite customer_data table.",
        len(df)
    )

    return engine, len(df)


def verify_load(engine):
    """Verify records stored in the database."""

    with engine.connect() as connection:
        count = connection.execute(
            text("SELECT COUNT(*) FROM customer_data")
        ).scalar()

        sample = connection.execute(
            text("""
                SELECT customer_id, age, monthly_spend,
                       contract_type, churn
                FROM customer_data
                LIMIT 5
            """)
        ).fetchall()

    return count, sample


if __name__ == "__main__":
    engine, loaded_records = load_data(
        "data/processed/customer_data_transformed.csv"
    )

    count, sample_records = verify_load(engine)

    print("\nETL LOAD PHASE SUCCESSFUL")
    print("-------------------------")
    print(f"Records loaded: {loaded_records}")
    print(f"Records verified in SQLite: {count}")
    print("Database: data/customer_database.db")
    print("Table: customer_data")

    print("\nSample database records:")

    for record in sample_records:
        print(record)
