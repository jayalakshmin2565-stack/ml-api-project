import asyncio
import time
from pathlib import Path

import httpx


URL = "http://127.0.0.1:8000/api/v1/predict"
TOTAL_REQUESTS = 100


def get_api_key():
    env_file = Path(".env")

    for line in env_file.read_text().splitlines():
        if line.startswith("API_KEY="):
            return line.split("=", 1)[1].strip()

    raise RuntimeError("API_KEY not found in .env")


API_KEY = get_api_key()


async def send_request(client):
    payload = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }

    headers = {
        "X-API-Key": API_KEY
    }

    start = time.perf_counter()

    try:
        response = await client.post(
            URL,
            json=payload,
            headers=headers
        )

        duration = time.perf_counter() - start

        return response.status_code, duration

    except Exception as exc:
        duration = time.perf_counter() - start

        return f"ERROR: {exc}", duration


async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:

        start = time.perf_counter()

        results = await asyncio.gather(
            *[
                send_request(client)
                for _ in range(TOTAL_REQUESTS)
            ]
        )

        total_time = time.perf_counter() - start

    successful = sum(
        1
        for status, _ in results
        if status == 200
    )

    failed = TOTAL_REQUESTS - successful

    durations = [
        duration
        for _, duration in results
    ]

    average_time = sum(durations) / len(durations)

    print()
    print("===== LOAD TEST RESULT =====")
    print(f"Total requests : {TOTAL_REQUESTS}")
    print(f"Successful     : {successful}")
    print(f"Failed         : {failed}")
    print(f"Total time     : {total_time:.3f} seconds")
    print(f"Average time   : {average_time:.4f} seconds")
    print("============================")


if __name__ == "__main__":
    asyncio.run(main())