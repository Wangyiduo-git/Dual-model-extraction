# Battery Data Extraction from PDF Literature

This Python script automates the extraction of battery experimental data from PDF files using AI models. It employs a two-stage approach: first classifying whether a document is battery-related, then extracting detailed experimental conditions.

## Features

- **Automated PDF Processing**: Supports both PyPDF2 and PyMuPDF for robust PDF text extraction
- **Two-Stage AI Pipeline**: 
  - Classification model to identify battery-related documents
  - Extraction model to extract detailed experimental conditions
- **OpenAI-Compatible API**: Works with DashScope and ModelScope APIs
- **Comprehensive Statistics**: Tracks processing time, token usage, and error rates
- **JSON Output**: Structured data output for easy integration
- **Retry Mechanism**: Automatic retry for failed API calls

## Requirements

```
pip install PyPDF2
pip install PyMuPDF
pip install openai
```

## Usage

### Command Line Arguments

```bash
python extract_battery_data.py [options]
```

**Required Arguments:**
- `--dashscope_url`: DashScope API URL for classification
- `--dashscope_key`: DashScope API key
- `--dashscope_model`: DashScope model name
- `--modelscope_url`: ModelScope API URL for extraction
- `--modelscope_key`: ModelScope API key

**Optional Arguments:**
- `--folder_path`: Folder path containing PDF files (if not provided, will prompt)
- `--modelscope_model`: ModelScope model name (default: deepseek-ai/DeepSeek-R1)

### Example Usage

```bash
python extract_battery_data.py \
  --folder_path "/path/to/pdfs" \
  --dashscope_url "https://dashscope.aliyuncs.com/api/v1" \
  --dashscope_key "your_dashscope_key" \
  --dashscope_model "qwen-max" \
  --modelscope_url "https://api.modelscope.cn/api/v1" \
  --modelscope_key "your_modelscope_key" \
  --modelscope_model "deepseek-ai/DeepSeek-R1"
```

## Output

### Extracted Data Format

The script extracts the following battery experimental conditions in JSON format:

```json
[
  {
    "DOI": "Literature DOI",
    "Battery Model": "Battery model",
    "DC Capacity (mAh/g)": "Discharge capacity",
    "Lower Voltage Limit (V)": "Lower voltage limit",
    "Upper Voltage Limit (V)": "Upper voltage limit",
    "Cathode Material": "Cathode material",
    "Anode Material": "Anode material",
    "Temperature (°C)": "Experimental temperature",
    "Charge Rate (C)": "Charge rate",
    "Discharge Rate (C)": "Discharge rate",
    "Capacity Retention (%)": "Capacity retention",
    "Cycle Count": "Cycle count",
    "Binder": "Binder",
    "Manufacturing Process": "Electrode manufacturing process"
  }
]
```

### Output Files

1. **Individual JSON files**: One per PDF file (`filename_conditions.json`)
2. **Processing statistics**: `processing_stats.json` with detailed metrics

## How It Works

1. **PDF Reading**: The script reads PDF files using PyMuPDF (primary) and PyPDF2 (fallback)
2. **Classification**: Uses the first AI model to determine if the document is battery-related
3. **Extraction**: For battery-related documents, uses the second AI model to extract experimental conditions
4. **Output**: Saves extracted data as structured JSON files

## Statistics and Monitoring

The script provides comprehensive statistics including:

- Total files processed
- Number of battery-related files
- Successfully extracted files
- API call times and token usage
- Error rates and retry statistics

## Error Handling

- Automatic retry mechanism for failed API calls (up to 3 attempts)
- Graceful handling of PDF reading failures
- Comprehensive error logging and statistics

## API Compatibility

This script uses the OpenAI API format, making it compatible with:
- DashScope (Alibaba Cloud)
- ModelScope
- Any OpenAI-compatible API endpoint

###Local model configuration
This tool now supports the use of locally deployed models without relying on external API services. To use the local model, please follow these steps:
1. Ensure that you have deployed model services that support OpenAI compatible interfaces locally (such as llama.cpp, vLLM, etc.)
2. Specify the local model service URL in the command line parameters, for example:
```bash
--dashscope_url " http://localhost:8000/v1 "
--modelscope_url " http://localhost:8000/v1 "
```
3. If your local model service does not require an API key, it can be set to any value:
```bash
--dashscope_key "not-needed"
--modelscope_key "not-needed"
```
For more information about local model deployment, please refer to the 'local_madels. md' document in the code repository.

## Support

If you have any questions or need assistance, please contact us through the following methods:

- e-mail：yiduo9132@gmai.com
