import time
import os
import PyPDF2
import json
import re
import fitz  # PyMuPDF
import traceback
import argparse
import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Any
from openai import OpenAI  # OpenAI-style API

# Command line argument parsing
parser = argparse.ArgumentParser(description='Extract battery data from PDF files using AI models.')
parser.add_argument('--folder_path', type=str, help='Folder path containing PDF literature')

# API related parameters
parser.add_argument('--dashscope_url', type=str, default='http://localhost:8000/v1',
                    help='DashScope or local model API URL')
parser.add_argument('--dashscope_key', type=str, default='not-needed',
                    help='DashScope API key or any value for local model')
parser.add_argument('--modelscope_url', type=str, default='http://localhost:8000/v1',
                    help='ModelScope or local model API URL')
parser.add_argument('--modelscope_key', type=str, default='not-needed',
                    help='ModelScope API key or any value for local model')

# Model related parameters
parser.add_argument('--dashscope_model', type=str, default='local-model',
                    help='DashScope model name or local model name')
parser.add_argument('--modelscope_model', type=str, default='local-model',
                    help='ModelScope model name or local model name')

# Local model parameters
parser.add_argument('--use_local_model', action='store_true', help='Use local models instead of remote APIs')
parser.add_argument('--local_model_path', type=str, default='', help='Path to local model if not using API')

args = parser.parse_args()


# Advanced entity extractor class
class BatteryEntityExtractor:
    """
    Advanced extractor for standardizing key entities and parameters from battery literature.
    Integrates deep learning methods with domain-specific heuristic rules.
    """

    def __init__(self):
        # Temperature pattern recognition rules
        self.temperature_patterns = [
            # Standard temperature expressions, e.g., "25 °C", "25°C", "25 degrees Celsius"
            r'(\d+\.?\d*)\s*(?:°|deg(?:rees?)?|℃)?\s*[Cc](?:\b|elsius)',
            # Celsius with subscript notation, e.g., "25℃"
            r'(\d+\.?\d*)\s*℃',
            # Room temperature expressions, e.g., "room temperature", "RT", "ambient temperature"
            r'(?:room|ambient)\s+temp(?:erature)?|(?<!\w)RT(?!\w)',
            # Negative temperatures, e.g., "-20 °C", "−10°C"
            r'(?:-|−)(\d+\.?\d*)\s*(?:°|deg(?:rees?)?|℃)?\s*[Cc](?:\b|elsius)',
            # Temperature ranges, e.g., "25-30°C", "25~30°C", "25 to 30°C", "between 25°C and 30°C"
            r'(\d+\.?\d*)\s*(?:-|–|—|~|to|and)\s*(\d+\.?\d*)\s*(?:°|deg(?:rees?)?|℃)?\s*[Cc](?:\b|elsius)'
        ]

        # Voltage pattern recognition rules
        self.voltage_patterns = [
            # Single voltage values, e.g., "4.2 V", "4.2V", "4.2 volts"
            r'(\d+\.?\d*)\s*(?:V|volts?)(?!\w)',
            # Voltage ranges, e.g., "2.5-4.2 V", "2.5~4.2V", "2.5 to 4.2 V"
            r'(\d+\.?\d*)\s*(?:-|–|—|~|to|and)\s*(\d+\.?\d*)\s*(?:V|volts?)(?!\w)',
            # Cut-off voltages, e.g., "cut-off voltage of 3.0 V", "cut-off: 3.0V"
            r'(?:cut[\s-]*off|cutoff|limiting)(?:\s+voltage)?(?:\s+of)?(?:\s*:)?\s*(\d+\.?\d*)\s*(?:V|volts?)(?!\w)',
            # Upper/lower limit voltages, e.g., "upper limit: 4.2V", "lower limit: 2.5V"
            r'(?:upper|lower|high|low)(?:\s+limit|\s+cutoff|\s+voltage)(?:\s+of)?(?:\s*:)?\s*(\d+\.?\d*)\s*(?:V|volts?)(?!\w)',
            # Voltage windows, e.g., "voltage window: 2.5-4.2V"
            r'voltage\s+window(?:\s+of)?(?:\s*:)?\s*(\d+\.?\d*)\s*(?:-|–|—|~|to|and)\s*(\d+\.?\d*)\s*(?:V|volts?)(?!\w)'
        ]

        # C-rate pattern recognition rules
        self.crate_patterns = [
            # Standard C-rates, e.g., "0.5C", "0.5 C", "C/2", "0.5 C-rate"
            r'(\d+\.?\d*)\s*[Cc](?:\-?rate)?(?!\w)|[Cc][/-](\d+\.?\d*)(?!\w)',
            # Asymmetric charge/discharge rates, e.g., "1C/2C", "0.5C charge/1C discharge"
            r'(\d+\.?\d*)\s*[Cc](?:\-?rate)?(?!\w)?\s*[/-]\s*(\d+\.?\d*)\s*[Cc](?:\-?rate)?(?!\w)?',
            # Explicitly specified charge/discharge rates, e.g., "0.2C charge, 1C discharge"
            r'(\d+\.?\d*)\s*[Cc](?:\-?rate)?(?!\w)?\s+(?:charge|charg\.|charging)(?:\s+and|\s*,)?\s+(\d+\.?\d*)\s*[Cc](?:\-?rate)?(?!\w)?\s+(?:discharge|disch\.|discharging)',
            # Current density expressions, e.g., "200 mA g-1", "200 mA/g", "200mA g^-1"
            r'(\d+\.?\d*)\s*(?:m|micro|μ|u)?[Aa](?:/|\s*g(?:ram)?[-−]?1|(?:\s*g(?:ram)?[\^]?[-−]?1))',
            # Constant current expressions, e.g., "constant current 0.5C charge/discharge"
            r'constant\s+current\s+(?:of\s+)?(\d+\.?\d*)\s*[Cc](?:\-?rate)?(?!\w)?'
        ]

        # Capacity pattern recognition rules
        self.capacity_patterns = [
            # Standard capacity expressions, e.g., "150 mAh/g", "150 mAh g-1", "150mAh g^-1"
            r'(\d+\.?\d*)\s*(?:m|micro|μ|u)?[Aa]h(?:/|\s*g(?:ram)?[-−]?1|(?:\s*g(?:ram)?[\^]?[-−]?1))',
            # Capacity units in mAh/g, e.g., "capacity of 150 mAh/g"
            r'(?:capacity|specific capacity)(?:\s+of)?\s*:?\s*(\d+\.?\d*)\s*(?:m|micro|μ|u)?[Aa]h(?:/|\s*g(?:ram)?[-−]?1|(?:\s*g(?:ram)?[\^]?[-−]?1))',
            # Capacity retention, e.g., "capacity retention of 85%", "85% capacity retention"
            r'(?:capacity\s+retention|retention\s+of\s+capacity)(?:\s+of)?\s*:?\s*(\d+\.?\d*)\s*%|(\d+\.?\d*)\s*%\s+(?:capacity\s+retention|retention\s+of\s+capacity)'
        ]

        # Cycle count pattern recognition rules
        self.cycle_patterns = [
            # Cycle counts, e.g., "100 cycles", "after 100 cycles", "100th cycle"
            r'(?:for|after)\s+(\d+)\s+cycles?|(\d+)(?:th|st|nd|rd)?\s+cycles?|cycles?(?:\s+of)?\s*:?\s*(\d+)',
            # Cycle life, e.g., "cycle life of 500 cycles"
            r'cycle\s+life(?:\s+of)?\s*:?\s*(\d+)(?:\s+cycles?)?'
        ]

        # Electrode material recognition rules
        self.material_patterns = {
            'cathode': [
                # Layered oxides, e.g., "LiNi0.8Co0.1Mn0.1O2", "NCM811", "NCA"
                r'Li(?:thium)?Ni(?:ckel)?[\d\.]*Co(?:balt)?[\d\.]*(?:Mn(?:ganese)?|Al(?:uminum)?)[\d\.]*O[\d\.]*',
                r'NC(?:M|A)(?:\d{3}|\d{4}|\d{6})',
                r'Li(?:thium)?(?:Ni(?:ckel)?|Co(?:balt)?|Mn(?:ganese)?|Al(?:uminum)?|Fe(?:rrum)?|P(?:hosphate)?|O(?:xygen)?|S(?:ulfur)?)+',
                # Olivine structures, e.g., "LiFePO4", "LFP"
                r'LiFePO[\d\.]*|LFP(?!\w)',
                # Spinel structures, e.g., "LiMn2O4", "LMO"
                r'LiMn[\d\.]*O[\d\.]*|LMO(?!\w)',
                # Research-specific names, e.g., "LNMCO", "Sample A"
                r'L[A-Z]{2,5}O|Sample\s+[A-Z]'
            ],
            'anode': [
                # Graphite, e.g., "graphite", "mesocarbon microbeads"
                r'graph(?:ite|ene)|carbon|mesocarbon\s+microbeads|MCMB',
                # Silicon-carbon composites, e.g., "Si/C", "Silicon-carbon composite"
                r'Si(?:licon)?(?:/|-|\s+and\s+|\+)C(?:arbon)?|Si(?:licon)?-C(?:arbon)?\s+composite',
                # Lithium metal, e.g., "lithium metal", "Li metal"
                r'li(?:thium)?\s+metal',
                # Lithium titanate, e.g., "Li4Ti5O12", "LTO"
                r'Li[\d\.]+Ti[\d\.]+O[\d\.]+|LTO(?!\w)'
            ]
        }

        # Binder recognition rules
        self.binder_patterns = [
            # Common binders, e.g., "PVDF", "CMC", "SBR"
            r'(?:P|poly)(?:VDF|vinylidene\s+fluoride)',
            r'(?:C|carboxy)(?:MC|methyl\s+cellulose)',
            r'(?:S|styrene)(?:BR|butadiene\s+rubber)',
            r'(?:PAA|poly\s*acrylic\s+acid)',
            r'(?:PTFE|polytetrafluoroethylene)'
        ]

        # Electrolyte pattern recognition rules
        self.electrolyte_patterns = [
            # Liquid electrolytes, e.g., "1M LiPF6 in EC/DMC (1:1)"
            r'(\d+\.?\d*)\s*M\s+Li[A-Z]+[\d]*\s+in\s+[A-Z/]+(?:\s+\(\d+:\d+\))?',
            # General electrolyte descriptions, e.g., "liquid electrolyte", "carbonate-based electrolyte"
            r'(?:liquid|solid|gel|carbonate[\-\s]based)\s+electrolyte'
        ]

    def extract_temperature(self, text: str) -> Union[float, str, None]:
        """Extract and standardize temperature values"""
        for pattern in self.temperature_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Special handling for room temperature
                if "room" in match.group().lower() or "ambient" in match.group().lower() or "rt" in match.group().lower():
                    return 25.0  # Standard room temperature defined as 25°C

                # Handle temperature ranges
                if re.search(r'\d+\s*(?:-|–|—|~|to|and)\s*\d+', match.group()):
                    temps = re.findall(r'(\d+\.?\d*)', match.group())
                    if len(temps) >= 2:
                        temp1, temp2 = float(temps[0]), float(temps[1])
                        return (temp1 + temp2) / 2  # Return average value

                # Handle negative temperatures
                if "-" in match.group() or "−" in match.group():
                    neg_temps = re.findall(r'(?:-|−)(\d+\.?\d*)', match.group())
                    if neg_temps:
                        return -float(neg_temps[0])

                # Handle standard temperature values
                temps = re.findall(r'(\d+\.?\d*)', match.group())
                if temps:
                    return float(temps[0])

        return None

    def extract_voltage_limits(self, text: str) -> Dict[str, Optional[float]]:
        """Extract and standardize voltage upper and lower limits"""
        result = {"lower": None, "upper": None}

        # First look for explicit upper/lower limit expressions
        lower_match = re.search(
            r'(?:lower|low)(?:\s+limit|\s+cutoff|\s+voltage)(?:\s+of)?(?:\s*:)?\s*(\d+\.?\d*)\s*(?:V|volts?)', text,
            re.IGNORECASE)
        upper_match = re.search(
            r'(?:upper|high)(?:\s+limit|\s+cutoff|\s+voltage)(?:\s+of)?(?:\s*:)?\s*(\d+\.?\d*)\s*(?:V|volts?)', text,
            re.IGNORECASE)

        if lower_match:
            result["lower"] = float(lower_match.group(1))

        if upper_match:
            result["upper"] = float(upper_match.group(1))

        # Look for voltage range expressions
        for pattern in self.voltage_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if re.search(r'\d+\s*(?:-|–|—|~|to|and)\s*\d+', match.group()):
                    volts = re.findall(r'(\d+\.?\d*)', match.group())
                    if len(volts) >= 2:
                        v1, v2 = float(volts[0]), float(volts[1])
                        result["lower"] = min(v1, v2)
                        result["upper"] = max(v1, v2)
                        break

        # Look for cut-off voltage expressions
        cutoff_match = re.search(
            r'(?:cut[\s-]*off|cutoff|limiting)(?:\s+voltage)?(?:\s+of)?(?:\s*:)?\s*(\d+\.?\d*)\s*(?:V|volts?)', text,
            re.IGNORECASE)
        if cutoff_match:
            # Cut-off voltage typically refers to lower limit
            if not result["lower"]:
                result["lower"] = float(cutoff_match.group(1))

        return result

    def extract_crates(self, text: str) -> Dict[str, Optional[float]]:
        """Extract and standardize charge/discharge rates"""
        result = {"charge": None, "discharge": None}

        # First check for explicit charge/discharge rate distinctions
        # Handle cases like "0.2C charge, 1C discharge"
        charge_discharge_match = re.search(
            r'(\d+\.?\d*)\s*[Cc](?:\-?rate)?(?!\w)?\s+(?:charge|charg\.|charging)(?:\s+and|\s*,)?\s+(\d+\.?\d*)\s*[Cc](?:\-?rate)?(?!\w)?\s+(?:discharge|disch\.|discharging)',
            text, re.IGNORECASE)
        if charge_discharge_match:
            result["charge"] = self._normalize_crate(charge_discharge_match.group(1))
            result["discharge"] = self._normalize_crate(charge_discharge_match.group(2))
            return result

        # Handle cases like "1C/2C"
        asymmetric_match = re.search(
            r'(\d+\.?\d*)\s*[Cc](?:\-?rate)?(?!\w)?\s*[/-]\s*(\d+\.?\d*)\s*[Cc](?:\-?rate)?(?!\w)?', text,
            re.IGNORECASE)
        if asymmetric_match:
            result["charge"] = self._normalize_crate(asymmetric_match.group(1))
            result["discharge"] = self._normalize_crate(asymmetric_match.group(2))
            return result

        # Handle constant current cases, e.g., "constant current 0.5C charge/discharge"
        constant_current_match = re.search(r'constant\s+current\s+(?:of\s+)?(\d+\.?\d*)\s*[Cc](?:\-?rate)?(?!\w)?',
                                           text, re.IGNORECASE)
        if constant_current_match:
            rate = self._normalize_crate(constant_current_match.group(1))
            result["charge"] = rate
            result["discharge"] = rate
            return result

        # Handle general cases without explicit charge/discharge distinction
        for pattern in self.crate_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Handle C/n format
                if match.group() and '/' in match.group() and match.group().startswith('C'):
                    denominator = re.search(r'[Cc][/-](\d+\.?\d*)', match.group())
                    if denominator and denominator.group(1):
                        rate = 1.0 / float(denominator.group(1))
                        result["charge"] = rate
                        result["discharge"] = rate
                        return result

                # Handle standard nC format
                values = re.findall(r'(\d+\.?\d*)', match.group())
                if values:
                    rate = self._normalize_crate(values[0])
                    result["charge"] = rate
                    result["discharge"] = rate
                    return result

        return result

    def _normalize_crate(self, rate_str: str) -> float:
        """Standardize C-rate values"""
        try:
            return float(rate_str)
        except:
            return None

    def extract_capacity(self, text: str) -> Optional[float]:
        """Extract and standardize discharge capacity"""
        for pattern in self.capacity_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values = re.findall(r'(\d+\.?\d*)', match.group())
                if values:
                    return float(values[0])
        return None

    def extract_capacity_retention(self, text: str) -> Optional[float]:
        """Extract and standardize capacity retention"""
        retention_match = re.search(
            r'(?:capacity\s+retention|retention\s+of\s+capacity)(?:\s+of)?\s*:?\s*(\d+\.?\d*)\s*%|(\d+\.?\d*)\s*%\s+(?:capacity\s+retention|retention\s+of\s+capacity)',
            text, re.IGNORECASE)
        if retention_match:
            groups = retention_match.groups()
            for g in groups:
                if g:
                    return float(g)
        return None

    def extract_cycle_count(self, text: str) -> Optional[int]:
        """Extract and standardize cycle count"""
        for pattern in self.cycle_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                for g in groups:
                    if g and g.isdigit():
                        return int(g)
        return None

    def identify_material(self, text: str, material_type: str) -> Optional[str]:
        """Identify electrode materials"""
        if material_type not in self.material_patterns:
            return None

        for pattern in self.material_patterns[material_type]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group().strip()

        return None

    def identify_binder(self, text: str) -> Optional[str]:
        """Identify binder materials"""
        for pattern in self.binder_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group().strip()
        return None

    def standardize_json_data(self, json_data: List[Dict], full_text: str = "") -> List[Dict]:
        """
        Standardize all entities and parameters in JSON data
        Integrates advanced pattern recognition and semantic parsing
        """
        for item in json_data:
            # Add complete text to each item for deep entity recognition
            if full_text:
                item["raw_text"] = full_text

            # Temperature standardization
            if "Temperature (°C)" in item:
                if isinstance(item["Temperature (°C)"], str) and item["Temperature (°C)"] != "N/A":
                    extracted_temp = self.extract_temperature(item["Temperature (°C)"])
                    if extracted_temp is not None:
                        item["Temperature (°C)"] = extracted_temp
                    elif full_text:
                        # Try to extract from full text
                        extracted_temp = self.extract_temperature(full_text)
                        if extracted_temp is not None:
                            item["Temperature (°C)"] = extracted_temp

            # Voltage limits standardization
            if "Lower Voltage Limit (V)" in item or "Upper Voltage Limit (V)" in item:
                # Try to extract from item data
                voltage_text = ""
                if "Lower Voltage Limit (V)" in item and item["Lower Voltage Limit (V)"]:
                    voltage_text += str(item["Lower Voltage Limit (V)"]) + " "
                if "Upper Voltage Limit (V)" in item and item["Upper Voltage Limit (V)"]:
                    voltage_text += str(item["Upper Voltage Limit (V)"]) + " "

                # If item data is insufficient, extract from full text
                if full_text and (not voltage_text or voltage_text.strip() == "N/A N/A "):
                    voltage_limits = self.extract_voltage_limits(full_text)
                    if voltage_limits["lower"] is not None:
                        item["Lower Voltage Limit (V)"] = voltage_limits["lower"]
                    if voltage_limits["upper"] is not None:
                        item["Upper Voltage Limit (V)"] = voltage_limits["upper"]
                else:
                    # Extract from item data
                    voltage_limits = self.extract_voltage_limits(voltage_text)
                    if voltage_limits["lower"] is not None and "Lower Voltage Limit (V)" in item:
                        item["Lower Voltage Limit (V)"] = voltage_limits["lower"]
                    if voltage_limits["upper"] is not None and "Upper Voltage Limit (V)" in item:
                        item["Upper Voltage Limit (V)"] = voltage_limits["upper"]

            # Charge/discharge rates standardization
            if "Charge Rate (C)" in item or "Discharge Rate (C)" in item:
                # Try to extract from item data
                rate_text = ""
                if "Charge Rate (C)" in item and item["Charge Rate (C)"]:
                    rate_text += "charge: " + str(item["Charge Rate (C)"]) + " "
                if "Discharge Rate (C)" in item and item["Discharge Rate (C)"]:
                    rate_text += "discharge: " + str(item["Discharge Rate (C)"])

                # If item data is insufficient, extract from full text
                if full_text and (not rate_text or "N/A" in rate_text):
                    crates = self.extract_crates(full_text)
                    if crates["charge"] is not None and "Charge Rate (C)" in item:
                        item["Charge Rate (C)"] = crates["charge"]
                    if crates["discharge"] is not None and "Discharge Rate (C)" in item:
                        item["Discharge Rate (C)"] = crates["discharge"]
                else:
                    # Handle "1C/2C" format
                    asymmetric_match = re.search(r'(\d+\.?\d*)\s*[Cc]\s*[/-]\s*(\d+\.?\d*)\s*[Cc]', rate_text)
                    if asymmetric_match:
                        if "Charge Rate (C)" in item:
                            item["Charge Rate (C)"] = float(asymmetric_match.group(1))
                        if "Discharge Rate (C)" in item:
                            item["Discharge Rate (C)"] = float(asymmetric_match.group(2))

                    # Handle standard C-rates
                    if "Charge Rate (C)" in item and isinstance(item["Charge Rate (C)"], str) and item[
                        "Charge Rate (C)"] != "N/A":
                        charge_match = re.search(r'(\d+\.?\d*)', item["Charge Rate (C)"])
                        if charge_match:
                            item["Charge Rate (C)"] = float(charge_match.group(1))

                    if "Discharge Rate (C)" in item and isinstance(item["Discharge Rate (C)"], str) and item[
                        "Discharge Rate (C)"] != "N/A":
                        discharge_match = re.search(r'(\d+\.?\d*)', item["Discharge Rate (C)"])
                        if discharge_match:
                            item["Discharge Rate (C)"] = float(discharge_match.group(1))

            # Discharge capacity standardization
            if "DC Capacity (mAh/g)" in item:
                if isinstance(item["DC Capacity (mAh/g)"], str) and item["DC Capacity (mAh/g)"] != "N/A":
                    capacity = self.extract_capacity(item["DC Capacity (mAh/g)"])
                    if capacity is not None:
                        item["DC Capacity (mAh/g)"] = capacity
                    elif full_text:
                        # Try to extract from full text
                        capacity = self.extract_capacity(full_text)
                        if capacity is not None:
                            item["DC Capacity (mAh/g)"] = capacity

            # Capacity retention standardization
            if "Capacity Retention (%)" in item:
                if isinstance(item["Capacity Retention (%)"], str) and item["Capacity Retention (%)"] != "N/A":
                    retention = self.extract_capacity_retention(item["Capacity Retention (%)"])
                    if retention is not None:
                        item["Capacity Retention (%)"] = retention
                    elif full_text:
                        # Try to extract from full text
                        retention = self.extract_capacity_retention(full_text)
                        if retention is not None:
                            item["Capacity Retention (%)"] = retention

            # Cycle count standardization
            if "Cycle Count" in item:
                if isinstance(item["Cycle Count"], str) and item["Cycle Count"] != "N/A":
                    cycles = self.extract_cycle_count(item["Cycle Count"])
                    if cycles is not None:
                        item["Cycle Count"] = cycles
                    elif full_text:
                        # Try to extract from full text
                        cycles = self.extract_cycle_count(full_text)
                        if cycles is not None:
                            item["Cycle Count"] = cycles

            # Electrode material identification
            if "Cathode Material" in item and (
                    item["Cathode Material"] == "N/A" or not item["Cathode Material"]) and full_text:
                cathode = self.identify_material(full_text, "cathode")
                if cathode:
                    item["Cathode Material"] = cathode

            if "Anode Material" in item and (
                    item["Anode Material"] == "N/A" or not item["Anode Material"]) and full_text:
                anode = self.identify_material(full_text, "anode")
                if anode:
                    item["Anode Material"] = anode

            # Binder identification
            if "Binder" in item and (item["Binder"] == "N/A" or not item["Binder"]) and full_text:
                binder = self.identify_binder(full_text)
                if binder:
                    item["Binder"] = binder

            # Remove temporary raw text field
            if "raw_text" in item:
                del item["raw_text"]

        return json_data


# Initialize clients with OpenAI API format
# Client for classification model (DashScope or local)
dashscope_client = OpenAI(
    base_url=args.dashscope_url,
    api_key=args.dashscope_key,
)

# Client for extraction model (ModelScope or local)
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


# Print model configuration information
def print_model_config():
    print("\n===== Model Configuration Information =====")
    if args.use_local_model:
        print("Mode: Local Model")
        print(f"Local model path: {args.local_model_path if args.local_model_path else 'Using default path'}")
    else:
        print("Mode: Remote API")
        print(f"Classification model (DashScope) URL: {args.dashscope_url}")
        print(f"Extraction model (ModelScope) URL: {args.modelscope_url}")

    print(f"Classification model name: {args.dashscope_model}")
    print(f"Extraction model name: {args.modelscope_model}")
    print("==============================================\n")


# Read PDF content using PyPDF2
def read_pdf_with_pypdf2(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            # Only read the first few pages to get the abstract (usually in the first two pages)
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
            # Only read the first few pages to get the abstract (usually in the first two pages)
            max_pages = min(2, len(pdf_document))
            for page_num in range(max_pages):
                text += pdf_document[page_num].get_text("text")
        return text
    except Exception as e:
        print(f"❌ PyMuPDF PDF reading failed: {os.path.basename(file_path)}, Error: {e}")
        return None


# Extract abstract section
def extract_abstract(text):
    # Common abstract identifier words
    abstract_keywords = [
        "abstract", "summary",
        "highlights", "graphical abstract"
    ]

    # Common sections that follow the abstract
    end_keywords = [
        "introduction", "keywords", "1.", "i.",
        "experimental"
    ]

    # First try to find the start of the abstract
    start_index = -1
    for keyword in abstract_keywords:
        pattern = re.compile(rf'{keyword}[\s]*[:]?', re.IGNORECASE)
        match = pattern.search(text)
        if match:
            start_index = match.end()
            break

    # If no abstract marker is found, use the first 1000 characters as the abstract
    if start_index == -1:
        return text[:1000]

    # Try to find the end of the abstract
    end_index = len(text)
    for keyword in end_keywords:
        pattern = re.compile(rf'\n\s*{keyword}[\s]*[:]?', re.IGNORECASE)
        match = pattern.search(text[start_index:])
        if match:
            end_index = start_index + match.start()
            break

    # Extract abstract text
    abstract = text[start_index:end_index].strip()

    # If the extracted abstract is too long, limit its length
    if len(abstract) > 2000:
        abstract = abstract[:2000]

    return abstract


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

    # Initialize entity extractor
    entity_extractor = BatteryEntityExtractor()

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

        # Extract abstract section
        abstract_text = extract_abstract(pdf_text)
        print(f"Extracted abstract length: {len(abstract_text)} characters")

        # Prepare prompt for first model (classification) - only using abstract content
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

        # Prepare prompt for second model (extraction) - using complete PDF text for detailed information extraction
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
        For charge/discharge rates in the format "1C/2C", specify that the first value (1C) is for charging and the second value (2C) is for discharging.
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

            # Use advanced entity extractor for data standardization
            standardized_json_data = entity_extractor.standardize_json_data(json_data, pdf_text)

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
    # Print model configuration information
    print_model_config()

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