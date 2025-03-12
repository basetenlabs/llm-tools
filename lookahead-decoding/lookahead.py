import math
import os
import json
import requests
import time
import aiohttp
import asyncio
from tqdm import tqdm
import argparse
import pandas as pd
import io
import matplotlib.pyplot as plt


def max_draft_len(windows_size, ngram_size, verification_set_size):
    return (0 if (ngram_size == 1) else ngram_size - 2) + (windows_size - 1 + verification_set_size) * (ngram_size - 1)


async def fetch(session, url, headers, json):
    start_time = time.time()
    async with session.post(url, headers=headers, json=json) as response:
        return time.time() - start_time


async def main(model_id, api_key, requests_dir, max_requests, concurrency, engine_window_size, engine_ngram_size, engine_verification_set):

    model_url = f"https://model-{model_id}.api.baseten.co/environments/production/predict"

    request_data = []

    # Sweeps
    window_sizes = list(range(1, engine_window_size, 1))
    verification_set_sizes = list(range(1, engine_verification_set, 1))
    ngram_sizes = list(pow(2, i)
                       for i in range(1, int(math.sqrt(engine_ngram_size))-1))

    if engine_ngram_size not in ngram_sizes:
        ngram_sizes.append(engine_ngram_size)

    # Sending any value that exceeds this will result in an error
    engine_max_draft_len = max_draft_len(
        engine_window_size, engine_ngram_size, engine_verification_set)

    # Load all requests from the requests directory
    for i, filename in enumerate(os.listdir(requests_dir)):
        if i >= max_requests:
            break
        with open(os.path.join(requests_dir, filename), 'r') as file:
            if not filename.endswith('.json'):
                continue
            data = json.load(file)
            request_data.append(data)
            
    print(f"Found {len(request_data)} requests from {requests_dir}")

    # Send a request to the model to warm it up
    print("Running a warm up request to ensure the model is ready...")
    async with aiohttp.ClientSession() as session:
        await fetch(session, model_url, {"Authorization": f"Api-Key {api_key}"}, request_data[0])

    results = []

    for window_size in window_sizes:
        for ngram_size in ngram_sizes:
            for verification_set_size in verification_set_sizes:

                sweep_key = f"{window_size},{ngram_size},{verification_set_size}"

                print(
                    f"Running with window_size: {window_size}, ngram_size: {ngram_size}, verification_set_size: {verification_set_size}")

                max_draft_len_value = max_draft_len(
                    window_size, ngram_size, verification_set_size)

                if (max_draft_len_value > engine_max_draft_len):
                    print(
                        f"Skipping window_size: {window_size}, ngram_size: {ngram_size}, verification_set_size: {verification_set_size} as it exceeds the engine's max draft length")
                    continue

                latencies = []

                semaphore = asyncio.Semaphore(concurrency)

                async def controlled_request(request):
                    async with semaphore:
                        # Add lookahead decoding config to the request
                        request["lookahead_decoding_config"] = {
                            "window_size": window_size,
                            "ngram_size": ngram_size,
                            "verification_set_size": verification_set_size
                        }
                        try:
                            async with aiohttp.ClientSession() as session:
                                latency = await fetch(session, model_url, {"Authorization": f"Api-Key {api_key}"}, request)
                                latencies.append(latency)
                        except:
                            print(
                                f"Error with window_size: {window_size}, ngram_size: {ngram_size}, verification_set_size: {verification_set_size}")

                tasks = []

                for request in request_data:
                    tasks.append(asyncio.create_task(
                        controlled_request(request)))

                with tqdm(range(len(tasks))) as progress_bar:
                    for task in asyncio.as_completed(tasks):
                        await task
                        progress_bar.update(1)

                latencies = sorted(latencies)

                results.append(
                    {
                        'Run': sweep_key,
                        'Avg': sum(latencies)/len(latencies),
                        'Max': max(latencies),
                        'Min': min(latencies),
                        'P50': latencies[int(len(latencies) * 0.5)],
                        'P90': latencies[int(len(latencies) * 0.9)],
                        'P99': latencies[int(len(latencies) * 0.99)]
                    }
                )

    df = pd.DataFrame(results)
    process_results(df)
    plot_results(df)


def process_results(df):
    min_avg = df['Avg'].idxmin()
    min_p50 = df['P50'].idxmin()
    min_p90 = df['P90'].idxmin()
    min_p99 = df['P99'].idxmin()

    print(
        f"Best Avg: {df.loc[min_avg, 'Run']} at {df.loc[min_avg, 'Avg']} seconds")
    print(
        f"Best P50:  {df.loc[min_p50, 'Run']} at {df.loc[min_p50, 'P50']} seconds")
    print(
        f"Best P90:  {df.loc[min_p90, 'Run']} at {df.loc[min_p90, 'P90']} seconds")
    print(
        f"Best P99:  {df.loc[min_p99, 'Run']} at {df.loc[min_p99, 'P99']} seconds")


def plot_results(df):

    df.plot(y=['P50', 'P90', 'P99'], x='Run')
    plt.xticks(range(len(df['Run'])), df['Run'], rotation=90)
    plt.title('Latency vs Lookahead Decoding Config')
    plt.xlabel(
        'Lookahead Decoding Config (window_size, ngram_size, verification_set_size)')
    plt.show()


if __name__ == "__main__":
    msg = "Lookahead decoding performance sweeps to determine the optimal settings for a given TRT model"
    parser = argparse.ArgumentParser(description=msg)

    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help='The model URL to send requests to')

    parser.add_argument('--api_key',
                        type=str,
                        required=True,
                        help='The baseten API key for the model')

    parser.add_argument('--requests_dir',
                        type=str,
                        required=False,
                        default='./requests',
                        help='The directory containing the requests to send to the model. Each request should be a separate JSON file')

    parser.add_argument('--max_requests',
                        type=int,
                        required=False,
                        default=10000000,
                        help='The maximum number of requests to use from the request_dir. This is useful for running a subset of the requests')

    parser.add_argument('--request_concurrency',
                        type=int,
                        required=False,
                        default=10,
                        help='The maximum number of parallel requests to send to the model')

    parser.add_argument('--window_size',
                        type=int,
                        required=True,
                        help='The maximum window size for the lookahead decoding. This should match the configuration the engine was built with')

    parser.add_argument('--ngram_size',
                        type=int,
                        required=True,
                        help='The maximum ngram size for the lookahead decoding. This should match the configuration the engine was built with')

    parser.add_argument('--verification_size',
                        type=int,
                        required=True,
                        help='The maximum ngram size for the lookahead decoding. This should match the configuration the engine was built with')

    args = parser.parse_args()

    asyncio.run(main(args.model, args.api_key, args.requests_dir, args.max_requests, args.request_concurrency, args.window_size,
                args.ngram_size, args.verification_size))
