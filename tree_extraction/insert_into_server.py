import requests
import json
import os
import time
import argparse
import traceback
import datetime

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help = "The path to the directory of results we want to upload to the server")
    return parser.parse_args()

def make_requests():
    request_url = "http://127.0.0.1:9543/record_run"
    args = read_args()
    for file_name in os.listdir(args.input_dir):
        if "json" not in file_name or file_name[0] == '.':
            continue
        
        # Read in the current file
        with open(os.path.join(args.input_dir, file_name), "r") as reader:
            request_data = json.load(reader)

        # Make the request
        request_data["run_id"] = "run_" + str(abs(hash(file_name))) + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f"))
        try:
            start_time = time.time()
            response = requests.post(url = request_url, json = request_data)
            response_value = response.json()
            if "error" in response_value:
                raise Exception("SERVER ERROR - " + response_value["error"])
            time_taken = round(1000.0 * (time.time() - start_time))

            print("Processed file", file_name, "sucessfully in", time_taken, "ms with response", response_value)
        except:
            print("FAIL: Request for file", file_name, "due to error", traceback.format_exc())
        
if __name__ == "__main__":
    make_requests()