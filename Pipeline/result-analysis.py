import os
import json
import matplotlib.pyplot as plt

def process_json_files(input_dir):
    total_token_counts = []
    processing_times = []

    # Iterate through files in the directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_dir, file_name)
            with open(file_path, 'r') as file:
                data = json.load(file)

                # Extract relevant fields
                total_token_count = data.get("metadata", {}).get("totalTokenCount", 0)
                processing_time = data.get("processing-time", 0)

                total_token_counts.append(total_token_count)
                processing_times.append(processing_time)

    # Calculate the sums
    sum_token_counts = sum(total_token_counts)
    sum_processing_times = sum(processing_times)

    total_token_counts.sort()
    processing_times.sort()

    # Print results
    print(f"Maximum token count: {total_token_counts[-1]}")
    print(f"Minimum token count: {total_token_counts[0]}")

    print(f"Maximum processing time: {processing_times[-1]}")
    print(f"Minimum processing time: {processing_times[0]}")

    print(f"Total token counts: {sum_token_counts}")
    print(f"Total processing times: {sum_processing_times}")

    result_file_path = os.path.join(input_dir, "analysis.txt")
    with open(result_file_path, "w") as result_file:
        result_file.write(f"Maximum token count: {total_token_counts[-1]}\n")
        result_file.write(f"Minimum token count: {total_token_counts[0]}\n")

        result_file.write(f"Maximum processing time: {processing_times[-1]}\n")
        result_file.write(f"Minimum processing time: {processing_times[0]}\n")

        result_file.write(f"Total token count: {sum_token_counts}\n")
        result_file.write(f"Total processing time: {sum_processing_times}\n")


    # Save plots
    save_plots(total_token_counts, processing_times, input_dir)


def save_plots(total_token_counts, processing_times, input_dir):
    # Plot for token counts
    plt.figure()
    plt.bar(range(len(total_token_counts)), total_token_counts)
    # plt.xlabel("File Index")
    plt.ylabel("Token Count")
    plt.title("Token Counts of Each File")
    plt.savefig(os.path.join(input_dir, "token_counts_plot.png"))

    # Plot for processing times
    plt.figure()
    plt.bar(range(len(processing_times)), processing_times)
    # plt.xlabel("File Index")
    plt.ylabel("Processing Time (s)")
    plt.title("Processing Times of Each File")
    plt.savefig(os.path.join(input_dir, "processing_times_plot.png"))


# Replace 'input_directory_path' with your actual input directory path
input_directory_path = "outputs/gemini-flash-inContext-structured"
process_json_files(input_directory_path)
