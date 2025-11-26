import argparse, requests, time, os, json, subprocess
from requests.exceptions import HTTPError, RequestException


parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, required=True)
parser.add_argument("--tenant_id", required=True)
parser.add_argument("--ontology_id", required=True)
parser.add_argument("--token", required=True)
parser.add_argument("--result_path", required=True)
parser.add_argument("--cdn_url", required=True)
args = parser.parse_args()

# Read the token from the file
with open(args.token, 'r') as f:
    token_value = f.read().strip()

# Calculate epoch time range (last 1 hour) - DYNAMIC TIMESTAMPS
end_time = int(time.time() * 1000)
start_time = end_time - (60 * 60 * 1000)

print(f"Fetching data from {start_time} to {end_time}")

url = "https://igs.gov-cloud.ai/mobius-graph-operations-test/cypher/execute-cypher"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token_value}"
}

# Pagination logic - fetch all records in batches of 100
all_data_items = []
skip = 0
batch_size = args.limit

while True:
    # Updated cypher query with dynamic timestamps and pagination
    # cypher_query = f'MATCH (n)-[r]-(m) WHERE (n.PI_314_CREATIONTIMEMS >= {start_time} AND n.PI_314_CREATIONTIMEMS <= {end_time}) OR (m.PI_314_CREATIONTIMEMS >= {start_time} AND m.PI_314_CREATIONTIMEMS <= {end_time}) WITH n, r, m SKIP {skip} LIMIT {batch_size} WITH COLLECT([{{elementId: elementId(n), identity: id(n), labels: labels(n), properties: properties(n)}}, {{elementId: elementId(r), identity: id(r), type: type(r), start: id(startNode(r)), end: id(endNode(r)), properties: properties(r)}}, {{elementId: elementId(m), identity: id(m), labels: labels(m), properties: properties(m)}}]) AS data RETURN ["n", "r", "m"] AS columns, data'
    # cypher_query: f'MATCH (n)-[r]-(m) WHERE (n.PI_314_CREATIONTIMEMS >= 1760351731270 AND n.PI_314_CREATIONTIMEMS <= 1760351731272) OR (m.PI_314_CREATIONTIMEMS >= 1760351731270 AND m.PI_314_CREATIONTIMEMS <= 1760351731272) WITH n, r, m LIMIT 100 WITH COLLECT([{elementId: elementId(n), identity: id(n), labels: labels(n), properties: properties(n)}, {elementId: elementId(r), identity: id(r), type: type(r), start: id(startNode(r)), end: id(endNode(r)), properties: properties(r)}, {elementId: elementId(m), identity: id(m), labels: labels(m), properties: properties(m)}]) AS data RETURN [\"n\", \"r\", \"m\"] AS columns, data'
    # cypher_query = "MATCH (n)-[r]-(m) WHERE (n.PI_314_CREATIONTIMEMS >= 1760351731270 AND n.PI_314_CREATIONTIMEMS <= 1760351731272) OR (m.PI_314_CREATIONTIMEMS >= 1760351731270 AND m.PI_314_CREATIONTIMEMS <= 1760351731272) WITH n, r, m LIMIT 100 WITH COLLECT([{elementId: elementId(n), identity: id(n), labels: labels(n), properties: properties(n)}, {elementId: elementId(r), identity: id(r), type: type(r), start: id(startNode(r)), end: id(endNode(r)), properties: properties(r)}, {elementId: elementId(m), identity: id(m), labels: labels(m), properties: properties(m)}]) AS data RETURN ['n', 'r', 'm'] AS columns, data"

    # cypher_query = 'MATCH (n)-[r]-(m) ' \
    #     'WHERE (n.PI_314_CREATIONTIMEMS >= ' + str(start_time) + ' AND n.PI_314_CREATIONTIMEMS <= ' + str(end_time) + ') ' \
    #     'OR (m.PI_314_CREATIONTIMEMS >= ' + str(start_time) + ' AND m.PI_314_CREATIONTIMEMS <= ' + str(end_time) + ') ' \
    #     'WITH n, r, m ' \
    #     'SKIP ' + str(skip) + ' LIMIT ' + str(batch_size) + ' ' \
    #     'WITH COLLECT([{elementId: elementId(n), identity: id(n), labels: labels(n), properties: properties(n)}, ' \
    #     '{elementId: elementId(r), identity: id(r), type: type(r), start: id(startNode(r)), end: id(endNode(r)), properties: properties(r)}, ' \
    #     '{elementId: elementId(m), identity: id(m), labels: labels(m), properties: properties(m)}]) AS data ' \
    #     'RETURN ["n", "r", "m"] AS columns, data' 

    # cypher_query = f'MATCH (n)-[r]->(m) WHERE (n.PI_314_CREATIONTIMEMS >= {start_time} AND n.PI_314_CREATIONTIMEMS <= {end_time}) OR (m.PI_314_CREATIONTIMEMS >= {start_time} AND m.PI_314_CREATIONTIMEMS <= {end_time}) WITH n, r, m SKIP {skip} LIMIT {batch_size} WITH COLLECT([{{elementId: elementId(n), identity: id(n), labels: labels(n), properties: properties(n)}}, {{elementId: elementId(r), identity: id(r), type: type(r), start: id(startNode(r)), end: id(endNode(r)), properties: properties(r)}}, {{elementId: elementId(m), identity: id(m), labels: labels(m), properties: properties(m)}}]) AS data RETURN ["n", "r", "m"] AS columns, data'

    cypher_query = f'MATCH (n)-[r]->(m) WITH n, r, m SKIP {skip} LIMIT {batch_size} WITH COLLECT([{{elementId: elementId(n), identity: id(n), labels: labels(n), properties: properties(n)}}, {{elementId: elementId(r), identity: id(r), type: type(r), start: id(startNode(r)), end: id(endNode(r)), properties: properties(r)}}, {{elementId: elementId(m), identity: id(m), labels: labels(m), properties: properties(m)}}]) AS data RETURN ["n", "r", "m"] AS columns, data'


    payload = {
        "cypher_query": cypher_query.strip(),
        "tenant_id": args.tenant_id,
        # "ontology_id": args.ontology_id,
        "ontology_id": "",
        "db_name": "AgentsProd"
    }

    # print(f"Fetching batch with SKIP {skip}, LIMIT {batch_size}...")
    
    # response = requests.post(url, headers=headers, json=payload)
    # response.raise_for_status()
    # response_data = response.json()
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            print(f"Fetching batch with SKIP {skip}, LIMIT {batch_size}... (Attempt {retry_count + 1}/{max_retries})")
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            response_data = response.json()
            
            # If successful, break out of retry loop
            break
            
        except (RequestException, requests.exceptions.HTTPError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Failed after {max_retries} attempts. Error: {e}")
                raise
            else:
                wait_time = 2 ** retry_count  # Exponential backoff: 2, 4, 8 seconds
                print(f"Request failed with error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)



    #########################################################3
    
    # Extract the actual data array from the nested structure
    batch_results = response_data.get("result", [])
    
    if not batch_results or len(batch_results) == 0:
        print(f"No result structure found. Total records fetched: {len(all_data_items)}")
        break
    
    # Get the data array from inside the result object
    batch_data = batch_results[0].get("data", [])
    
    if not batch_data or len(batch_data) == 0:
        print(f"No more data found in batch. Total records fetched: {len(all_data_items)}")
        break
    
    all_data_items.extend(batch_data)
    print(f"Fetched {len(batch_data)} data items. Total so far: {len(all_data_items)}")
    
    # If we got less than batch_size, we've reached the end
    if len(batch_data) < batch_size:
        print(f"Reached end of results. Total records: {len(all_data_items)}")
        break
    
    skip += batch_size

# Create the final result structure with all combined data
final_result = [
    {
        "columns": ["n", "r", "m"],
        "data": all_data_items
    }
]

# Save all combined results to file
os.makedirs(args.result_path, exist_ok=True)
result_path_dir = os.path.join(args.result_path, 'result.json')
with open(result_path_dir, "w") as f:
    json.dump(final_result, f, indent=2)

print(f"Saved {len(final_result)} total records to {result_path_dir}")

upload_url = "https://igs.gov-cloud.ai/mobius-content-service/v1.0/content/upload?filePathAccess=private&filePath=%2Fbottle%2Flimka%2Fsoda%2F"

def upload_file_to_cdn(file_path, filename_prefix):
    # The filename for the CDN will be derived from the original file path
    original_filename = os.path.basename(file_path)
    cdn_filename = f"{filename_prefix}_{original_filename}"

    print(f"Uploading file from {file_path} with CDN name {cdn_filename} to {upload_url}")

    curl_command = [
        "curl",
        "--location",
        upload_url,
        "--header", f"Authorization: Bearer {token_value}",
        "--form", f"file=@{file_path}",  # Pass file path directly
        "--fail",
        "--show-error"
    ]

    print(f"Executing curl command: {' '.join(curl_command)}")

    try:
        process = subprocess.run(
            curl_command,
            capture_output=True,
            check=True  # Keep check=True to raise CalledProcessError
        )

        print("Upload successful. Raw response:")
        print(process.stdout.decode('utf-8'))  # Decode stdout for printing

        response_json = json.loads(process.stdout.decode('utf-8'))
        relative_cdn_url = response_json.get("cdnUrl", "URL_NOT_FOUND")

        if relative_cdn_url == "URL_NOT_FOUND":
            print("Error: Could not find 'cdnUrl' in the server response.")
            print("Full response:", process.stdout.decode('utf-8'))
            raise ValueError("Failed to parse cdnUrl from CDN response.")

        content_url_value = f"https://cdn-new.gov-cloud.ai{relative_cdn_url}"
        print(f"Extracted CDN URL: {content_url_value}")

        return content_url_value

    except subprocess.CalledProcessError as e:
        print("Error executing curl command:")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.stdout.decode('utf-8')}")
        print(f"Error Output: {e.stderr.decode('utf-8')}")
        raise e
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error processing the server response: {e}")
        raise e




cdn_url = upload_file_to_cdn(result_path_dir, "result")
print(f"Final Config CDN URL: {cdn_url}")

# Create directory and write cdn_url to output file
os.makedirs(os.path.dirname(args.cdn_url), exist_ok=True)
with open(args.cdn_url, "w") as f:
    f.write(cdn_url)







# parser = argparse.ArgumentParser()
# parser.add_argument("--limit", type=int, required=True)
# parser.add_argument("--tenant_id", required=True)
# parser.add_argument("--ontology_id", required=True)
# parser.add_argument("--token", required=True)
# parser.add_argument("--result_path", required=True)
# parser.add_argument("--cdn_url", required=True)
# args = parser.parse_args()

# # Read the token from the file
# with open(args.token, 'r') as f:
#     token_value = f.read().strip()

# # Calculate epoch time range (last 1 hour) - DYNAMIC TIMESTAMPS
# end_time = int(time.time() * 1000)
# start_time = end_time - (60 * 60 * 1000)

# print(f"Fetching data from {start_time} to {end_time}")

# url = "https://igs.gov-cloud.ai/mobius-graph-operations-test/cypher/execute-cypher"
# headers = {
#     "accept": "application/json",
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {token_value}"
# }

# # Pagination logic - fetch all records in batches
# all_data_items = []
# skip = 0
# batch_size = args.limit
# max_retries = 3

# while True:
#     cypher_query = f'MATCH (n)-[r]->(m) WITH n, r, m SKIP {skip} LIMIT {batch_size} WITH COLLECT([{{elementId: elementId(n), identity: id(n), labels: labels(n), properties: properties(n)}}, {{elementId: elementId(r), identity: id(r), type: type(r), start: id(startNode(r)), end: id(endNode(r)), properties: properties(r)}}, {{elementId: elementId(m), identity: id(m), labels: labels(m), properties: properties(m)}}]) AS data RETURN ["n", "r", "m"] AS columns, data'

#     payload = {
#         "cypher_query": cypher_query.strip(),
#         "tenant_id": args.tenant_id,
#         "ontology_id": "",
#         "db_name": "AgentsProd"
#     }

#     print(f"Fetching batch with SKIP {skip}, LIMIT {batch_size}...")
    
#     # Retry logic with exponential backoff
#     retry_delay = 2
#     response = None
    
#     for attempt in range(max_retries):
#         try:
#             response = requests.post(url, headers=headers, json=payload, timeout=60)
#             response.raise_for_status()
#             break  # Success, exit retry loop
            
#         except HTTPError as e:
#             if e.response.status_code == 500:
#                 if attempt < max_retries - 1:
#                     print(f"Server error (500), retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
#                     time.sleep(retry_delay)
#                     retry_delay *= 2  # Exponential backoff
#                 else:
#                     print(f"Failed after {max_retries} attempts at SKIP {skip}")
#                     print(f"Total records fetched before error: {len(all_data_items)}")
#                     print(f"Error response: {e.response.text}")
#                     # Save what we have so far
#                     if all_data_items:
#                         print("Saving partial results...")
#                         break
#                     else:
#                         raise
#             else:
#                 print(f"HTTP Error {e.response.status_code}: {e.response.text}")
#                 raise
                
#         except requests.exceptions.Timeout:
#             if attempt < max_retries - 1:
#                 print(f"Request timeout, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
#                 time.sleep(retry_delay)
#                 retry_delay *= 2
#             else:
#                 print(f"Timeout after {max_retries} attempts at SKIP {skip}")
#                 print(f"Total records fetched before timeout: {len(all_data_items)}")
#                 break
                
#         except requests.exceptions.RequestException as e:
#             print(f"Request error: {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(retry_delay)
#                 retry_delay *= 2
#             else:
#                 raise
    
#     # If we failed all retries, break the loop
#     if response is None or not response.ok:
#         print("Stopping pagination due to persistent errors.")
#         break
    
#     response_data = response.json()
    
#     # Extract the actual data array from the nested structure
#     batch_results = response_data.get("result", [])
    
#     if not batch_results or len(batch_results) == 0:
#         print(f"No result structure found. Total records fetched: {len(all_data_items)}")
#         break
    
#     # Get the data array from inside the result object
#     batch_data = batch_results[0].get("data", [])
    
#     if not batch_data or len(batch_data) == 0:
#         print(f"No more data found in batch. Total records fetched: {len(all_data_items)}")
#         break
    
#     all_data_items.extend(batch_data)
#     print(f"Fetched {len(batch_data)} data items. Total so far: {len(all_data_items)}")
    
#     # If we got less than batch_size, we've reached the end
#     if len(batch_data) < batch_size:
#         print(f"Reached end of results. Total records: {len(all_data_items)}")
#         break
    
#     skip += batch_size
    
#     # Add delay between successful requests to avoid rate limiting
#     time.sleep(0.5)

# # Check if we have any data to save
# if not all_data_items:
#     print("No data was fetched. Exiting without creating files.")
#     exit(1)

# # Create the final result structure with all combined data
# final_result = [
#     {
#         "columns": ["n", "r", "m"],
#         "data": all_data_items
#     }
# ]

# # Save all combined results to file
# os.makedirs(args.result_path, exist_ok=True)
# result_path_dir = os.path.join(args.result_path, 'result.json')
# with open(result_path_dir, "w") as f:
#     json.dump(final_result, f, indent=2)

# print(f"Saved {len(all_data_items)} total records to {result_path_dir}")

# upload_url = "https://igs.gov-cloud.ai/mobius-content-service/v1.0/content/upload?filePathAccess=private&filePath=%2Fbottle%2Flimka%2Fsoda%2F"

# def upload_file_to_cdn(file_path, filename_prefix):
#     # The filename for the CDN will be derived from the original file path
#     original_filename = os.path.basename(file_path)
#     cdn_filename = f"{filename_prefix}_{original_filename}"

#     print(f"Uploading file from {file_path} with CDN name {cdn_filename} to {upload_url}")

#     curl_command = [
#         "curl",
#         "--location",
#         upload_url,
#         "--header", f"Authorization: Bearer {token_value}",
#         "--form", f"file=@{file_path}",  # Pass file path directly
#         "--fail",
#         "--show-error"
#     ]

#     print(f"Executing curl command: {' '.join(curl_command)}")

#     try:
#         process = subprocess.run(
#             curl_command,
#             capture_output=True,
#             check=True  # Keep check=True to raise CalledProcessError
#         )

#         print("Upload successful. Raw response:")
#         print(process.stdout.decode('utf-8'))  # Decode stdout for printing

#         response_json = json.loads(process.stdout.decode('utf-8'))
#         relative_cdn_url = response_json.get("cdnUrl", "URL_NOT_FOUND")

#         if relative_cdn_url == "URL_NOT_FOUND":
#             print("Error: Could not find 'cdnUrl' in the server response.")
#             print("Full response:", process.stdout.decode('utf-8'))
#             raise ValueError("Failed to parse cdnUrl from CDN response.")

#         content_url_value = f"https://cdn-new.gov-cloud.ai{relative_cdn_url}"
#         print(f"Extracted CDN URL: {content_url_value}")

#         return content_url_value

#     except subprocess.CalledProcessError as e:
#         print("Error executing curl command:")
#         print(f"Return code: {e.returncode}")
#         print(f"Output: {e.stdout.decode('utf-8')}")
#         print(f"Error Output: {e.stderr.decode('utf-8')}")
#         raise e
#     except (json.JSONDecodeError, KeyError, ValueError) as e:
#         print(f"Error processing the server response: {e}")
#         raise e

# # Upload to CDN
# try:
#     cdn_url = upload_file_to_cdn(result_path_dir, "result")
#     print(f"Final Config CDN URL: {cdn_url}")

#     # Create directory and write cdn_url to output file
#     cdn_output_file = os.path.join(args.cdn_url, "cdn_url.txt")
#     os.makedirs(os.path.dirname(cdn_output_file), exist_ok=True)
#     with open(cdn_output_file, "w") as f:
#         f.write(cdn_url)
    
#     print(f"CDN URL written to: {cdn_output_file}")
    
# except Exception as e:
#     print(f"Failed to upload to CDN: {e}")
#     print("Local file is still available at:", result_path_dir)
#     exit(1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# import argparse, requests, time, os, json, subprocess

# parser = argparse.ArgumentParser()
# parser.add_argument("--limit", type=int, required=True)
# parser.add_argument("--tenant_id", required=True)
# parser.add_argument("--ontology_id", required=True)
# parser.add_argument("--token", required=True)
# parser.add_argument("--result_path", required=True)
# parser.add_argument("--cdn_url", required=True)
# args = parser.parse_args()

# # Read the token from the file
# with open(args.token, 'r') as f:
#     token_value = f.read().strip()

# # Calculate epoch time range (last 1 hour) - DYNAMIC TIMESTAMPS
# end_time = int(time.time() * 1000)
# start_time = end_time - (60 * 60 * 1000)

# print(f"Fetching data from {start_time} to {end_time}")

# url = "https://igs.gov-cloud.ai/mobius-graph-operations-test/cypher/execute-cypher"
# headers = {
#     "accept": "application/json",
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {token_value}"
# }

# # Pagination logic - fetch all records in batches of 100
# all_data_items = []
# skip = 0
# batch_size = args.limit

# while True:
#     # Updated cypher query with dynamic timestamps and pagination
#     # cypher_query = f'MATCH (n)-[r]-(m) WHERE (n.PI_314_CREATIONTIMEMS >= {start_time} AND n.PI_314_CREATIONTIMEMS <= {end_time}) OR (m.PI_314_CREATIONTIMEMS >= {start_time} AND m.PI_314_CREATIONTIMEMS <= {end_time}) WITH n, r, m SKIP {skip} LIMIT {batch_size} WITH COLLECT([{{elementId: elementId(n), identity: id(n), labels: labels(n), properties: properties(n)}}, {{elementId: elementId(r), identity: id(r), type: type(r), start: id(startNode(r)), end: id(endNode(r)), properties: properties(r)}}, {{elementId: elementId(m), identity: id(m), labels: labels(m), properties: properties(m)}}]) AS data RETURN ["n", "r", "m"] AS columns, data'
#     # cypher_query: f'MATCH (n)-[r]-(m) WHERE (n.PI_314_CREATIONTIMEMS >= 1760351731270 AND n.PI_314_CREATIONTIMEMS <= 1760351731272) OR (m.PI_314_CREATIONTIMEMS >= 1760351731270 AND m.PI_314_CREATIONTIMEMS <= 1760351731272) WITH n, r, m LIMIT 100 WITH COLLECT([{elementId: elementId(n), identity: id(n), labels: labels(n), properties: properties(n)}, {elementId: elementId(r), identity: id(r), type: type(r), start: id(startNode(r)), end: id(endNode(r)), properties: properties(r)}, {elementId: elementId(m), identity: id(m), labels: labels(m), properties: properties(m)}]) AS data RETURN [\"n\", \"r\", \"m\"] AS columns, data'
#     # cypher_query = "MATCH (n)-[r]-(m) WHERE (n.PI_314_CREATIONTIMEMS >= 1760351731270 AND n.PI_314_CREATIONTIMEMS <= 1760351731272) OR (m.PI_314_CREATIONTIMEMS >= 1760351731270 AND m.PI_314_CREATIONTIMEMS <= 1760351731272) WITH n, r, m LIMIT 100 WITH COLLECT([{elementId: elementId(n), identity: id(n), labels: labels(n), properties: properties(n)}, {elementId: elementId(r), identity: id(r), type: type(r), start: id(startNode(r)), end: id(endNode(r)), properties: properties(r)}, {elementId: elementId(m), identity: id(m), labels: labels(m), properties: properties(m)}]) AS data RETURN ['n', 'r', 'm'] AS columns, data"

#     # cypher_query = 'MATCH (n)-[r]-(m) ' \
#     #     'WHERE (n.PI_314_CREATIONTIMEMS >= ' + str(start_time) + ' AND n.PI_314_CREATIONTIMEMS <= ' + str(end_time) + ') ' \
#     #     'OR (m.PI_314_CREATIONTIMEMS >= ' + str(start_time) + ' AND m.PI_314_CREATIONTIMEMS <= ' + str(end_time) + ') ' \
#     #     'WITH n, r, m ' \
#     #     'SKIP ' + str(skip) + ' LIMIT ' + str(batch_size) + ' ' \
#     #     'WITH COLLECT([{elementId: elementId(n), identity: id(n), labels: labels(n), properties: properties(n)}, ' \
#     #     '{elementId: elementId(r), identity: id(r), type: type(r), start: id(startNode(r)), end: id(endNode(r)), properties: properties(r)}, ' \
#     #     '{elementId: elementId(m), identity: id(m), labels: labels(m), properties: properties(m)}]) AS data ' \
#     #     'RETURN ["n", "r", "m"] AS columns, data' 

#     # cypher_query = f'MATCH (n)-[r]->(m) WHERE (n.PI_314_CREATIONTIMEMS >= {start_time} AND n.PI_314_CREATIONTIMEMS <= {end_time}) OR (m.PI_314_CREATIONTIMEMS >= {start_time} AND m.PI_314_CREATIONTIMEMS <= {end_time}) WITH n, r, m SKIP {skip} LIMIT {batch_size} WITH COLLECT([{{elementId: elementId(n), identity: id(n), labels: labels(n), properties: properties(n)}}, {{elementId: elementId(r), identity: id(r), type: type(r), start: id(startNode(r)), end: id(endNode(r)), properties: properties(r)}}, {{elementId: elementId(m), identity: id(m), labels: labels(m), properties: properties(m)}}]) AS data RETURN ["n", "r", "m"] AS columns, data'

#     cypher_query = f'MATCH (n)-[r]->(m) WITH n, r, m SKIP {skip} LIMIT {batch_size} WITH COLLECT([{{elementId: elementId(n), identity: id(n), labels: labels(n), properties: properties(n)}}, {{elementId: elementId(r), identity: id(r), type: type(r), start: id(startNode(r)), end: id(endNode(r)), properties: properties(r)}}, {{elementId: elementId(m), identity: id(m), labels: labels(m), properties: properties(m)}}]) AS data RETURN ["n", "r", "m"] AS columns, data'


#     payload = {
#         "cypher_query": cypher_query.strip(),
#         "tenant_id": args.tenant_id,
#         # "ontology_id": args.ontology_id,
#         "ontology_id": "",
#         "db_name": "AgentsProd"
#     }

#     print(f"Fetching batch with SKIP {skip}, LIMIT {batch_size}...")
    
#     response = requests.post(url, headers=headers, json=payload)
#     response.raise_for_status()
#     response_data = response.json()
    
#     # Extract the actual data array from the nested structure
#     batch_results = response_data.get("result", [])
    
#     if not batch_results or len(batch_results) == 0:
#         print(f"No result structure found. Total records fetched: {len(all_data_items)}")
#         break
    
#     # Get the data array from inside the result object
#     batch_data = batch_results[0].get("data", [])
    
#     if not batch_data or len(batch_data) == 0:
#         print(f"No more data found in batch. Total records fetched: {len(all_data_items)}")
#         break
    
#     all_data_items.extend(batch_data)
#     print(f"Fetched {len(batch_data)} data items. Total so far: {len(all_data_items)}")
    
#     # If we got less than batch_size, we've reached the end
#     if len(batch_data) < batch_size:
#         print(f"Reached end of results. Total records: {len(all_data_items)}")
#         break
    
#     skip += batch_size

# # Create the final result structure with all combined data
# final_result = [
#     {
#         "columns": ["n", "r", "m"],
#         "data": all_data_items
#     }
# ]

# # Save all combined results to file
# os.makedirs(args.result_path, exist_ok=True)
# result_path_dir = os.path.join(args.result_path, 'result.json')
# with open(result_path_dir, "w") as f:
#     json.dump(final_result, f, indent=2)

# print(f"Saved {len(final_result)} total records to {result_path_dir}")

# upload_url = "https://igs.gov-cloud.ai/mobius-content-service/v1.0/content/upload?filePathAccess=private&filePath=%2Fbottle%2Flimka%2Fsoda%2F"

# def upload_file_to_cdn(file_path, filename_prefix):
#     # The filename for the CDN will be derived from the original file path
#     original_filename = os.path.basename(file_path)
#     cdn_filename = f"{filename_prefix}_{original_filename}"

#     print(f"Uploading file from {file_path} with CDN name {cdn_filename} to {upload_url}")

#     curl_command = [
#         "curl",
#         "--location",
#         upload_url,
#         "--header", f"Authorization: Bearer {token_value}",
#         "--form", f"file=@{file_path}",  # Pass file path directly
#         "--fail",
#         "--show-error"
#     ]

#     print(f"Executing curl command: {' '.join(curl_command)}")

#     try:
#         process = subprocess.run(
#             curl_command,
#             capture_output=True,
#             check=True  # Keep check=True to raise CalledProcessError
#         )

#         print("Upload successful. Raw response:")
#         print(process.stdout.decode('utf-8'))  # Decode stdout for printing

#         response_json = json.loads(process.stdout.decode('utf-8'))
#         relative_cdn_url = response_json.get("cdnUrl", "URL_NOT_FOUND")

#         if relative_cdn_url == "URL_NOT_FOUND":
#             print("Error: Could not find 'cdnUrl' in the server response.")
#             print("Full response:", process.stdout.decode('utf-8'))
#             raise ValueError("Failed to parse cdnUrl from CDN response.")

#         content_url_value = f"https://cdn-new.gov-cloud.ai{relative_cdn_url}"
#         print(f"Extracted CDN URL: {content_url_value}")

#         return content_url_value

#     except subprocess.CalledProcessError as e:
#         print("Error executing curl command:")
#         print(f"Return code: {e.returncode}")
#         print(f"Output: {e.stdout.decode('utf-8')}")
#         print(f"Error Output: {e.stderr.decode('utf-8')}")
#         raise e
#     except (json.JSONDecodeError, KeyError, ValueError) as e:
#         print(f"Error processing the server response: {e}")
#         raise e




# cdn_url = upload_file_to_cdn(result_path_dir, "result")
# print(f"Final Config CDN URL: {cdn_url}")

# # Create directory and write cdn_url to output file
# os.makedirs(os.path.dirname(args.cdn_url), exist_ok=True)
# with open(args.cdn_url, "w") as f:
#     f.write(cdn_url)




