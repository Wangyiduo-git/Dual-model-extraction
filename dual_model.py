import time
import os
import PyPDF2
import json
import re
import fitz  # PyMuPDF
import traceback
import argparse
from openai import OpenAI  # OpenAI-style API

# Command line argument parsing
parser = argparse.ArgumentParser(description='Extract battery data from PDF files using AI models.')
parser.add_argument('--folder_path', type=str, help='Folder path containing PDF literature')
parser.add_argument('--dashscope_url', type=str, default='', help='DashScope API URL')
parser.add_argument('--dashscope_key', type=str, default='', help='DashScope API key')
parser.add_argument('--modelscope_url', type=str, default='', help='ModelScope API URL')
parser.add_argument('--modelscope_key', type=str, default='', help='ModelScope API key')
parser.add_argument('--dashscope_model', type=str, default='', help='DashScope model name')
parser.add_argument('--modelscope_model', type=str, default='deepseek-ai/DeepSeek-R1', help='ModelScope model name')
args = parser.parse_args()

# Initialize clients with OpenAI API format
# Client for classification model (DashScope)
dashscope_client = OpenAI(
    base_url=args.dashscope_url,
    api_key=args.dashscope_key,
)

# Client for extraction model (ModelScope)
modelscope_client = OpenAI(
    base_url=args.modelscope_url,
    api_key=args.modelscope_key,
)


# Statistics class
class Stats:
    def __init__(self):
        self.total_files = 0
        self.processed_files = 0
        self.battery_related_files = 0
        self.successfully_extracted_files = 0

        # First model statistics
        self.total_first_model_time = 0
        self.total_first_model_input_tokens = 0
        self.total_first_model_output_tokens = 0

        # Second model statistics
        self.total_second_model_time = 0
        self.total_second_model_input_tokens = 0
        self.total_second_model_output_tokens = 0

        # Error statistics
        self.first_model_errors = 0
        self.second_model_errors = 0
        self.retry_successes = 0


# Initialize stats object
stats = Stats()


# Read PDF content using PyPDF2
def read_pdf_with_pypdf2(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            max_pages = min(2, len(reader.pages))
            for page_num in range(max_pages):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"❌ PyPDF2 PDF reading failed: {os.path.basename(file_path)}, Error: {e}")
        return None


# Read PDF content using PyMuPDF
def read_pdf_with_pymupdf(file_path):
    try:
        with fitz.open(file_path) as pdf_document:
            text = ""
            max_pages = min(2, len(pdf_document))
            for page_num in range(max_pages):
                text += pdf_document[page_num].get_text("text")
        return text
    except Exception as e:
        print(f"❌ PyMuPDF PDF reading failed: {os.path.basename(file_path)}, Error: {e}")
        return None


# 提取摘要部分
def extract_abstract(text):
    # 常见的摘要标识词
    abstract_keywords = [
        "abstract", "摘要", "summary",
        "highlights", "graphical abstract", "研究摘要"
    ]

    # 常见的摘要后面的部分
    end_keywords = [
        "introduction", "keywords", "1.", "i.", "关键词",
        "引言", "前言", "实验", "材料与方法", "experimental"
    ]

    # 先尝试找到摘要开始处
    start_index = -1
    for keyword in abstract_keywords:
        pattern = re.compile(rf'{keyword}[\s]*[:：]?', re.IGNORECASE)
        match = pattern.search(text)
        if match:
            start_index = match.end()
            break

    # 如果找不到摘要标记，就用前1000个字符作为摘要
    if start_index == -1:
        return text[:1000]

    end_index = len(text)
    for keyword in end_keywords:
        pattern = re.compile(rf'\n\s*{keyword}[\s]*[:：]?', re.IGNORECASE)
        match = pattern.search(text[start_index:])
        if match:
            end_index = start_index + match.start()
            break

    abstract = text[start_index:end_index].strip()

    if len(abstract) > 2000:
        abstract = abstract[:2000]

    return abstract


# 标准化数值函数
def standardize_value(value_str):
    if not value_str or value_str == "N/A":
        return "N/A"

    # 字符串类型转换
    if isinstance(value_str, (int, float)):
        return value_str

    try:
        # 处理范围值 (例如 "25-30℃" 或 "25~30℃")
        range_match = re.search(r'(\d+[\.,]?\d*)\s*[~\-–—]\s*(\d+[\.,]?\d*)', str(value_str))
        if range_match:
            val1 = float(range_match.group(1).replace(',', '.'))
            val2 = float(range_match.group(2).replace(',', '.'))
            return (val1 + val2) / 2  # 返回平均值

        # 处理单一数值 (移除单位如 "℃", "C", "%")
        value_match = re.search(r'(\d+[\.,]?\d*)', str(value_str))
        if value_match:
            return float(value_match.group(1).replace(',', '.'))

        return value_str
    except Exception as e:
        print(f"⚠️ 标准化值时出错: {value_str} - {e}")
        return value_str


# 标准化JSON数据
def standardize_json_data(json_data):
    for item in json_data:
        # 标准化温度
        if "Temperature (°C)" in item:
            item["Temperature (°C)"] = standardize_value(item["Temperature (°C)"])

        # 标准化充放电倍率
        if "Charge Rate (C)" in item:
            item["Charge Rate (C)"] = standardize_value(item["Charge Rate (C)"])

        if "Discharge Rate (C)" in item:
            item["Discharge Rate (C)"] = standardize_value(item["Discharge Rate (C)"])

        # 标准化容量保持率
        if "Capacity Retention (%)" in item:
            item["Capacity Retention (%)"] = standardize_value(item["Capacity Retention (%)"])

        # 标准化电压范围
        if "Lower Voltage Limit (V)" in item:
            item["Lower Voltage Limit (V)"] = standardize_value(item["Lower Voltage Limit (V)"])

        if "Upper Voltage Limit (V)" in item:
            item["Upper Voltage Limit (V)"] = standardize_value(item["Upper Voltage Limit (V)"])

        # 标准化放电容量
        if "DC Capacity (mAh/g)" in item:
            item["DC Capacity (mAh/g)"] = standardize_value(item["DC Capacity (mAh/g)"])

        # 标准化循环次数
        if "Cycle Count" in item:
            item["Cycle Count"] = standardize_value(item["Cycle Count"])

    return json_data


# Safely call model with retry mechanism
def call_model_safely(client, model, prompt, system_message="", max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"Model API call attempt {attempt + 1}/{max_retries}...")

            # Record start time
            start_time = time.time()

            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            if system_message:
                messages.insert(0, {"role": "system", "content": system_message})

            # Call model
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False
            )

            # Calculate call time
            end_time = time.time()
            call_duration = end_time - start_time

            # Basic validation
            if not hasattr(response, 'choices') or len(response.choices) == 0:
                print(f"Attempt {attempt + 1}: Response has no choices, retrying...")
                time.sleep(2)
                continue

            return response, call_duration

        except Exception as e:
            print(f"Attempt {attempt + 1} error: {e}")
            print(traceback.format_exc())
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("Maximum retry attempts reached, giving up.")

    # All retries failed
    return None, 0


# Process a single PDF file
def process_pdf(pdf_file_path, output_folder):
    print(f"\nProcessing file: {pdf_file_path}")
    file_basename = os.path.basename(pdf_file_path)

    try:
        # Read PDF content (prioritize PyMuPDF)
        pdf_text = read_pdf_with_pymupdf(pdf_file_path)

        # If PyMuPDF reading fails, try PyPDF2
        if pdf_text is None or not pdf_text.strip():
            print("Attempting to read file using PyPDF2...")
            pdf_text = read_pdf_with_pypdf2(pdf_file_path)

        # If both methods fail
        if not pdf_text or not pdf_text.strip():
            print(f"⚠️ PDF file {file_basename} parsed as empty, skipping")
            stats.processed_files += 1
            return

        # 提取摘要部分
        abstract_text = extract_abstract(pdf_text)
        print(f"Extracted abstract length: {len(abstract_text)} characters")

        # Prepare prompt for first model (classification) - 只使用摘要内容
        classification_prompt = f"""
        Please determine if the following text from a scientific paper abstract is related to battery life or capacity degradation. 
        If it is related to battery life or capacity degradation, please answer "Yes"; if not, please answer "No".

        Abstract content:
        {abstract_text}
        """

        # Call first model (classification)
        classification_response, classification_duration = call_model_safely(
            client=dashscope_client,
            model=args.dashscope_model,
            prompt=classification_prompt,
            system_message="You are an expert in identifying battery-related literature."
        )

        # Check if we got a valid response
        if classification_response is None:
            print(f"Failed to get response from classification model for {file_basename}")
            stats.first_model_errors += 1
            stats.processed_files += 1
            return

        # Update statistics (tokens might not be available in all APIs)
        stats.total_first_model_time += classification_duration
        try:
            stats.total_first_model_input_tokens += classification_response.usage.prompt_tokens
            stats.total_first_model_output_tokens += classification_response.usage.completion_tokens
            print(
                f"Classification model usage - Input tokens: {classification_response.usage.prompt_tokens}, Output tokens: {classification_response.usage.completion_tokens}")
        except AttributeError:
            print("Token usage information not available for classification model")

        print(f"Classification model call time: {classification_duration:.2f} seconds")

        # Get classification result
        is_battery_related = classification_response.choices[0].message.content.strip().lower()

        # Determine if related to battery life or capacity degradation
        if "no" in is_battery_related:
            print("Document not related to battery capacity degradation, stopping processing.")
            stats.processed_files += 1
            return

        # Update battery-related file count
        stats.battery_related_files += 1

        print("Document is battery-related, continuing to extract detailed information...")

        # Prepare prompt for second model (extraction) - 使用完整PDF文本进行详细信息提取
        extraction_prompt = f"""
        You are a battery experimental data extraction expert. Please extract the experimental conditions 
        from the following literature content and output in JSON format.
        Ensure the data is complete. If missing, please fill in "N/A".

        Literature content:
        {pdf_text}

        Output format (please strictly follow):
        ```json
        [
          {{
            "DOI": "<Literature DOI>",
            "Battery Model": "<Battery model>",
            "DC Capacity (mAh/g)": <Discharge capacity>,
            "Lower Voltage Limit (V)": <Lower voltage limit>,
            "Upper Voltage Limit (V)": <Upper voltage limit>,
            "Cathode Material": "<Cathode material>",
            "Anode Material": "<Anode material>",
            "Temperature (°C)": <Experimental temperature>,
            "Charge Rate (C)": <Charge rate>,
            "Discharge Rate (C)": <Discharge rate>,
            "Capacity Retention (%)": <Capacity retention>,
            "Cycle Count": <Cycle count>,
            "Binder": "<Binder>",
            "Manufacturing Process": "<Electrode manufacturing process>"
          }}
        ]
        ```

        Important: For numerical values, please provide only the numeric value without units. For example, for temperature write "25" not "25°C".
        For ranges like "25-30°C", calculate and provide the average value (27.5).
        Return only JSON format data, do not include any explanatory text.
        """

        # Call second model (extraction)
        extraction_response, extraction_duration = call_model_safely(
            client=modelscope_client,
            model=args.modelscope_model,
            prompt=extraction_prompt,
            system_message="You are an expert in extracting battery experimental data."
        )

        # Check if we got a valid response
        if extraction_response is None:
            print(f"Failed to get response from extraction model for {file_basename}")
            stats.second_model_errors += 1
            stats.processed_files += 1
            return

        # Update statistics
        stats.total_second_model_time += extraction_duration
        try:
            stats.total_second_model_input_tokens += extraction_response.usage.prompt_tokens
            stats.total_second_model_output_tokens += extraction_response.usage.completion_tokens
            print(
                f"Extraction model usage - Input tokens: {extraction_response.usage.prompt_tokens}, Output tokens: {extraction_response.usage.completion_tokens}")
        except AttributeError:
            print("Token usage information not available for extraction model")

        print(f"Extraction model call time: {extraction_duration:.2f} seconds")

        # Get extraction result
        response_content = extraction_response.choices[0].message.content

        # Parse JSON and save
        try:
            json_match = re.search(r"```json\s*([\s\S]+?)\s*```", response_content)
            cleaned_content = json_match.group(1) if json_match else response_content.strip()
            json_data = json.loads(cleaned_content)

            # 标准化JSON数据
            standardized_json_data = standardize_json_data(json_data)

            # Generate JSON filename
            base_name = os.path.splitext(file_basename)[0]
            json_filename = base_name + "_conditions.json"
            json_file_path = os.path.join(output_folder, json_filename)

            with open(json_file_path, "w", encoding="utf-8") as jsonfile:
                json.dump(standardized_json_data, jsonfile, indent=4, ensure_ascii=False)

            print(f"✅ JSON data saved to {json_file_path}")

            # Update successfully extracted file count
            stats.successfully_extracted_files += 1

        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {file_basename}, Error: {e}")
            print(f"⚠️ API returned content: {response_content}")
            stats.second_model_errors += 1

    except Exception as e:
        print(f"Error processing file {file_basename}: {e}")
        print(traceback.format_exc())
        stats.first_model_errors += 1

    # Update processed file count
    stats.processed_files += 1


# Print statistics
def print_stats(output_folder):
    print("\n========== Processing Statistics ==========")
    print(f"Total files: {stats.total_files}")
    print(f"Processed files: {stats.processed_files}")
    print(f"Battery-related files: {stats.battery_related_files}")
    print(f"Successfully extracted files: {stats.successfully_extracted_files}")

    # Error statistics
    print(f"\n----- Error Statistics -----")
    print(f"First model errors: {stats.first_model_errors}")
    print(f"Second model errors: {stats.second_model_errors}")
    print(f"Retry successes: {stats.retry_successes}")

    # First model statistics
    print("\n----- First Model Statistics -----")
    print(f"Total call time: {stats.total_first_model_time:.2f} seconds")
    if stats.processed_files > stats.first_model_errors:
        print(
            f"Average call time: {(stats.total_first_model_time / (stats.processed_files - stats.first_model_errors)):.2f} seconds/file")
    else:
        print("Average call time: No data")
    print(f"Total input tokens: {stats.total_first_model_input_tokens}")
    print(f"Total output tokens: {stats.total_first_model_output_tokens}")
    print(f"Total tokens consumed: {stats.total_first_model_input_tokens + stats.total_first_model_output_tokens}")

    # Second model statistics
    if stats.battery_related_files > 0:
        print("\n----- Second Model Statistics -----")
        print(f"Total call time: {stats.total_second_model_time:.2f} seconds")
        if stats.battery_related_files > stats.second_model_errors:
            print(
                f"Average call time: {(stats.total_second_model_time / (stats.battery_related_files - stats.second_model_errors)):.2f} seconds/file")
        else:
            print("Average call time: No data")
        print(f"Total input tokens: {stats.total_second_model_input_tokens}")
        print(f"Total output tokens: {stats.total_second_model_output_tokens}")
        print(
            f"Total tokens consumed: {stats.total_second_model_input_tokens + stats.total_second_model_output_tokens}")

    # Overall statistics
    print("\n----- Overall Statistics -----")
    total_time = stats.total_first_model_time + stats.total_second_model_time
    total_tokens = (stats.total_first_model_input_tokens + stats.total_first_model_output_tokens +
                    stats.total_second_model_input_tokens + stats.total_second_model_output_tokens)

    print(f"Total call time: {total_time:.2f} seconds")
    print(f"Total tokens consumed: {total_tokens}")

    # Save statistics to file
    stats_file = os.path.join(output_folder, "processing_stats.json")
    stats_data = {
        "total_files": stats.total_files,
        "processed_files": stats.processed_files,
        "battery_related_files": stats.battery_related_files,
        "successfully_extracted_files": stats.successfully_extracted_files,
        "error_stats": {
            "first_model_errors": stats.first_model_errors,
            "second_model_errors": stats.second_model_errors,
            "retry_successes": stats.retry_successes
        },
        "first_model": {
            "total_time": round(stats.total_first_model_time, 2),
            "average_time": round(stats.total_first_model_time / (stats.processed_files - stats.first_model_errors),
                                  2) if (stats.processed_files - stats.first_model_errors) > 0 else 0,
            "total_input_tokens": stats.total_first_model_input_tokens,
            "total_output_tokens": stats.total_first_model_output_tokens,
            "total_tokens": stats.total_first_model_input_tokens + stats.total_first_model_output_tokens
        },
        "second_model": {
            "total_time": round(stats.total_second_model_time, 2),
            "average_time": round(
                stats.total_second_model_time / (stats.battery_related_files - stats.second_model_errors), 2) if (
                                                                                                                         stats.battery_related_files - stats.second_model_errors) > 0 else 0,
            "total_input_tokens": stats.total_second_model_input_tokens,
            "total_output_tokens": stats.total_second_model_output_tokens,
            "total_tokens": stats.total_second_model_input_tokens + stats.total_second_model_output_tokens
        },
        "total": {
            "total_time": round(total_time, 2),
            "total_tokens": total_tokens
        }
    }

    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, indent=4, ensure_ascii=False)

    print(f"\nStatistics saved to: {stats_file}")


def main():
    # Get folder path from command line arguments or prompt user
    folder_path = args.folder_path
    if not folder_path:
        folder_path = input("Please enter the folder path containing PDF literature: ")

    # Create output folder
    output_folder = os.path.join(folder_path, "extraction_results")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all PDF files
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    # Update total file count
    stats.total_files = len(pdf_files)

    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
    else:
        print(f"Found {len(pdf_files)} PDF files, beginning processing...")

        # Process each PDF file
        for i, pdf_file in enumerate(pdf_files):
            print(f"\nProcessing progress: {i + 1}/{len(pdf_files)}")
            full_path = os.path.join(folder_path, pdf_file)
            process_pdf(full_path, output_folder)

        # Print final statistics
        print_stats(output_folder)

        print("\nAll files processed!")


if __name__ == "__main__":
    main()