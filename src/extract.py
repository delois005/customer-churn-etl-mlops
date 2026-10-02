import pandas as pd
import requests
import logging
from pathlib import Path

# Configure logging
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    filename="logs/etl_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def extract_csv(file_path, batch_size=5):
    """Extract CSV data in batches for scalable processing."""
    logging.info("Starting CSV extraction.")

    batches = []

    for batch in pd.read_csv(file_path, chunksize=batch_size):
        batches.append(batch)
        logging.info("Extracted batch containing %d records.", len(batch))

    data = pd.concat(batches, ignore_index=True)

    logging.info("CSV extraction completed: %d total records.", len(data))
    return data


def extract_api():
    """Extract sample unstructured/semi-structured data from a web API."""
    url = "https://jsonplaceholder.typicode.com/posts"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        api_data = response.json()

        logging.info(
            "API extraction completed: %d records received.",
            len(api_data)
        )

        return api_data[:5]

    except requests.RequestException as error:
        logging.error("API extraction failed: %s", error)
        return []


if __name__ == "__main__":
    csv_data = extract_csv("data/raw/customer_data.csv")
    api_data = extract_api()

    print("\nETL EXTRACT PHASE SUCCESSFUL")
    print("----------------------------")
    print(f"CSV records extracted: {len(csv_data)}")
    print(f"API records extracted: {len(api_data)}")
    print("\nSample structured data:")
    print(csv_data.head())

    if api_data:
        print("\nSample API data:")
        print(api_data[0])


def extract_database():
    """Extract structured data from SQLite using SQLAlchemy."""
    from sqlalchemy import create_engine, text

    engine = create_engine("sqlite:///data/customer_database.db")

    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT customer_id, age, monthly_spend,
                       contract_type, churn
                FROM customer_data
                LIMIT 5
            """)
        )

        rows = result.fetchall()

    logging.info(
        "Database extraction completed: %d records received.",
        len(rows)
    )

    return rows
