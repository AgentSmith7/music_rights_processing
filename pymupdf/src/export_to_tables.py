#!/usr/bin/env python3
"""
Export extracted JSON results to PostgreSQL-ready format.

Parses the JSON output from process_statements.py and generates:
1. PostgreSQL schema (DDL)
2. CSV files for COPY import
3. Metadata including confidence scores, bboxes, processing times
"""

import os
import sys
import json
import re
import csv
import uuid
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal, InvalidOperation

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ValueNormalizer:
    """Normalize extracted values to SQL-compatible types."""
    
    @staticmethod
    def parse_currency(value: str) -> Optional[Decimal]:
        """Parse currency string to Decimal. Handles $, (), negatives."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        
        value = str(value).strip()
        if not value or value.lower() in ('n/a', 'not specified', '-'):
            return None
        
        is_negative = False
        if value.startswith('(') and value.endswith(')'):
            is_negative = True
            value = value[1:-1]
        if value.startswith('-'):
            is_negative = True
            value = value[1:]
        if value.startswith('($') or value.startswith('-$'):
            is_negative = True
            value = value.replace('($', '').replace('-$', '').replace(')', '')
        
        value = value.replace('$', '').replace(',', '').strip()
        
        try:
            result = Decimal(value)
            return -result if is_negative else result
        except (InvalidOperation, ValueError):
            return None
    
    @staticmethod
    def parse_percentage(value: str) -> Optional[Decimal]:
        """Parse percentage string to Decimal (as fraction, e.g., 50% -> 0.50)."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return Decimal(str(value)) / 100
        
        value = str(value).strip()
        if not value or value.lower() in ('n/a', 'not specified', '-'):
            return None
        
        is_negative = False
        if value.startswith('-'):
            is_negative = True
            value = value[1:]
        
        value = value.replace('%', '').strip()
        
        try:
            result = Decimal(value) / 100
            return -result if is_negative else result
        except (InvalidOperation, ValueError):
            return None
    
    @staticmethod
    def parse_integer(value: str) -> Optional[int]:
        """Parse integer string."""
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        
        value = str(value).strip().replace(',', '')
        try:
            return int(value)
        except ValueError:
            return None
    
    @staticmethod
    def parse_date(value: str) -> Optional[str]:
        """Parse date string to ISO format."""
        if value is None:
            return None
        
        value = str(value).strip()
        if not value or value.lower() in ('n/a', 'not specified', '-'):
            return None
        
        date_formats = [
            '%m/%d/%Y',
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m-%d-%Y',
            '%B %d, %Y',
        ]
        
        for fmt in date_formats:
            try:
                parsed = datetime.strptime(value, fmt)
                return parsed.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return value
    
    @staticmethod
    def clean_string(value: str) -> Optional[str]:
        """Clean and normalize string value."""
        if value is None:
            return None
        value = str(value).strip()
        if not value or value.lower() in ('not specified', 'n/a'):
            return None
        return value


class TableDataExtractor:
    """Extract structured table data from raw_analysis JSON strings."""
    
    def __init__(self):
        self.normalizer = ValueNormalizer()
    
    def extract_json_from_markdown(self, text: str) -> Optional[Dict]:
        """Extract JSON object from markdown code blocks."""
        if not text:
            return None
        
        json_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
        matches = re.findall(json_pattern, text, re.MULTILINE)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    
    def extract_table_rows(self, raw_analysis: str) -> List[Dict]:
        """Extract table_data rows from raw_analysis - handles multiple LLM output formats."""
        parsed = self.extract_json_from_markdown(raw_analysis)
        if not parsed:
            return []
        
        table_data = []
        
        # Helper to normalize a row dict from NEW schema (line_items)
        def normalize_from_line_items(row: Dict) -> Dict:
            """Normalize from the new strict line_items schema."""
            return {
                'item_code': row.get('item_code'),
                'item_description': row.get('item_description'),
                'channel': row.get('channel'),
                'units': row.get('units'),
                'unit_price': row.get('unit_price'),
                'gross_amount': row.get('gross_amount'),
                'royalty_rate': row.get('royalty_rate'),
                'royalty_amount': row.get('royalty_amount'),
                '_schema': 'line_items'
            }
        
        # Helper to normalize a row dict from OLD schema (various formats)
        def normalize_row(row: Dict) -> Dict:
            """Normalize field names to standard format (legacy)."""
            normalized = {}
            # Territory/description
            normalized['territory'] = (
                row.get('territory') or 
                row.get('name') or 
                row.get('country')
            )
            # Platform/channel
            normalized['platform'] = (
                row.get('platform') or 
                row.get('stream_type') or
                row.get('channel')
            )
            # Sales units
            normalized['sales_units'] = (
                row.get('sales_units') or 
                row.get('units') or 
                row.get('units_sold')
            )
            # Unit price
            normalized['calc_per_unit_base'] = (
                row.get('calc_per_unit_base') or 
                row.get('calc_per_unit') or
                row.get('container_reduction') or
                row.get('royalty_rate')
            )
            # Royalty rate
            normalized['royalty_rate_percent'] = (
                row.get('royalty_rate_percent') or 
                row.get('royalty_rate_percentage') or
                row.get('royalty_rate')
            )
            # Royalty payable
            normalized['royalty_payable'] = (
                row.get('royalty_payable') or 
                row.get('royalty') or
                row.get('earnings')
            )
            normalized['_schema'] = 'legacy'
            return normalized
        
        # PRIORITY 1: Check for new "line_items" schema (strict format)
        if 'line_items' in parsed and isinstance(parsed['line_items'], list):
            for item in parsed['line_items']:
                if isinstance(item, dict):
                    table_data.append(normalize_from_line_items(item))
            if table_data:
                return table_data
        
        # 1. Check for table_data (primary location)
        if 'table_data' in parsed:
            data = parsed['table_data']
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        table_data.append(normalize_row(item))
            elif isinstance(data, dict):
                # Handle dict format with transactions or territories
                if 'transactions' in data:
                    for item in data.get('transactions', []):
                        if isinstance(item, dict):
                            table_data.append(normalize_row(item))
                elif 'territories' in data and isinstance(data['territories'], list):
                    # Check if it's parallel arrays or list of dicts
                    territories = data.get('territories', [])
                    if territories and isinstance(territories[0], dict):
                        # List of territory dicts
                        for item in territories:
                            table_data.append(normalize_row(item))
                    else:
                        # Parallel arrays format
                        calc_per_unit = data.get('calc_per_unit_base', [])
                        royalty_rate = data.get('royalty_rate', [])
                        sales_units = data.get('sales_units', [])
                        royalty_payable = data.get('royalty_payable', [])
                        
                        for i, territory in enumerate(territories):
                            table_data.append({
                                'territory': territory,
                                'calc_per_unit_base': calc_per_unit[i] if i < len(calc_per_unit) else None,
                                'royalty_rate_percent': royalty_rate[i] if i < len(royalty_rate) else None,
                                'sales_units': sales_units[i] if i < len(sales_units) else None,
                                'royalty_payable': royalty_payable[i] if i < len(royalty_payable) else None,
                            })
                else:
                    # Nested by stream type (e.g., stream_ad_supported, stream_subscription)
                    for stream_type, territories_data in data.items():
                        if isinstance(territories_data, dict):
                            for territory, values in territories_data.items():
                                if isinstance(values, dict):
                                    row = normalize_row(values)
                                    row['territory'] = territory
                                    row['platform'] = stream_type
                                    table_data.append(row)
        
        # 2. Check for detailed_table_data (alternative name)
        if not table_data and 'detailed_table_data' in parsed:
            data = parsed['detailed_table_data']
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        table_data.append(normalize_row(item))
        
        # 3. Check for royalty_rates_and_calculations
        if not table_data and 'royalty_rates_and_calculations' in parsed:
            data = parsed['royalty_rates_and_calculations']
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        table_data.append(normalize_row(item))
        
        # 4. Check nested in royalty_statement
        if not table_data and 'royalty_statement' in parsed:
            stmt = parsed['royalty_statement']
            if 'table_data' in stmt:
                data = stmt['table_data']
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            table_data.append(normalize_row(item))
        
        # 5. Check earnings_summaries.international.country_breakdown
        if not table_data and 'earnings_summaries' in parsed:
            intl = parsed['earnings_summaries'].get('international', {})
            if intl and isinstance(intl, dict) and 'country_breakdown' in intl:
                for item in intl.get('country_breakdown', []):
                    if isinstance(item, dict):
                        row = normalize_row(item)
                        row['territory'] = item.get('country')
                        table_data.append(row)
        
        # Filter out rows with territory = "--" or empty
        table_data = [
            row for row in table_data 
            if row.get('territory') and row.get('territory') not in ('--', 'Unknown', 'Not provided', '')
        ]
        
        return table_data
    
    def extract_summary_info(self, raw_analysis: str) -> Dict:
        """Extract summary/period info from raw_analysis."""
        parsed = self.extract_json_from_markdown(raw_analysis)
        if not parsed:
            return {}
        
        summary = {}
        
        if 'earnings_summaries' in parsed:
            summary['earnings'] = parsed['earnings_summaries']
        if 'earnings_summaries' in parsed.get('royalty_statement', {}):
            summary['earnings'] = parsed['royalty_statement']['earnings_summaries']
        
        for key in ['total_royalty_for_period', 'royalty_for_period']:
            if key in parsed:
                summary['royalty_for_period'] = parsed[key]
            if key in parsed.get('earnings_summaries', {}):
                summary['royalty_for_period'] = parsed['earnings_summaries'][key]
        
        if 'ending_balance' in parsed.get('earnings_summaries', {}):
            summary['ending_balance'] = parsed['earnings_summaries']['ending_balance']
        
        if 'vendor_artist_info' in parsed:
            summary['artist_info'] = parsed['vendor_artist_info']
        if 'vendor_artist_info' in parsed.get('royalty_statement', {}):
            summary['artist_info'] = parsed['royalty_statement']['vendor_artist_info']
        
        return summary
    
    def extract_expenses(self, raw_analysis: str) -> List[Dict]:
        """Extract expense/credit details from raw_analysis."""
        parsed = self.extract_json_from_markdown(raw_analysis)
        if not parsed:
            return []
        
        if 'table_data' in parsed and isinstance(parsed['table_data'], dict):
            return parsed['table_data'].get('expense_credit_detail', [])
        
        if 'payment_information' in parsed:
            payment = parsed['payment_information']
            if payment and isinstance(payment, dict) and payment.get('date') and payment.get('amount'):
                return [payment]
        
        return []


class PostgreSQLExporter:
    """Export extracted data to PostgreSQL-ready format."""
    
    SCHEMA_SQL = '''
-- Music Rights Royalty Statement Schema (Complete, No NULLs)
-- Generated by export_to_tables.py

-- Drop existing tables (in reverse dependency order)
DROP TABLE IF EXISTS royalty_line_items CASCADE;
DROP TABLE IF EXISTS statement_summaries CASCADE;
DROP TABLE IF EXISTS extraction_metadata CASCADE;
DROP TABLE IF EXISTS statements CASCADE;

-- Statements (one per PDF - document-level metadata)
CREATE TABLE statements (
    id UUID PRIMARY KEY,
    pdf_name VARCHAR(500) NOT NULL,
    vendor_name VARCHAR(255) NOT NULL DEFAULT 'N/A',
    contract_name VARCHAR(255) NOT NULL DEFAULT 'N/A',
    vendor_number VARCHAR(100) NOT NULL DEFAULT 'N/A',
    contract_number VARCHAR(100) NOT NULL DEFAULT 'N/A',
    period_start VARCHAR(50) NOT NULL DEFAULT 'N/A',
    period_end VARCHAR(50) NOT NULL DEFAULT 'N/A',
    total_pages INTEGER NOT NULL DEFAULT 0,
    processed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Statement summaries (earnings summaries, totals - NO NULLs)
CREATE TABLE statement_summaries (
    id UUID PRIMARY KEY,
    statement_id UUID NOT NULL REFERENCES statements(id),
    page_number INTEGER NOT NULL,
    category VARCHAR(255) NOT NULL,
    subcategory VARCHAR(255) NOT NULL DEFAULT 'N/A',
    amount DECIMAL(15,4) NOT NULL DEFAULT 0
);

-- Royalty line items (transaction-level detail - ALL fields required, NO NULLs)
CREATE TABLE royalty_line_items (
    id UUID PRIMARY KEY,
    statement_id UUID NOT NULL REFERENCES statements(id),
    page_number INTEGER NOT NULL,
    item_name VARCHAR(500) NOT NULL,
    item_code VARCHAR(100) NOT NULL DEFAULT 'N/A',
    channel VARCHAR(255) NOT NULL DEFAULT 'N/A',
    units INTEGER NOT NULL DEFAULT 0,
    unit_rate DECIMAL(15,6) NOT NULL DEFAULT 0,
    gross_amount DECIMAL(15,4) NOT NULL DEFAULT 0,
    royalty_rate DECIMAL(8,6) NOT NULL DEFAULT 0,
    royalty_amount DECIMAL(15,4) NOT NULL DEFAULT 0
);

-- Extraction metadata (for debugging/auditing)
CREATE TABLE extraction_metadata (
    id UUID PRIMARY KEY,
    statement_id UUID NOT NULL REFERENCES statements(id),
    page_number INTEGER NOT NULL,
    region_type VARCHAR(50) NOT NULL DEFAULT 'unknown',
    region_confidence DECIMAL(5,4) NOT NULL DEFAULT 0,
    bbox_x1 DECIMAL(10,2) NOT NULL DEFAULT 0,
    bbox_y1 DECIMAL(10,2) NOT NULL DEFAULT 0,
    bbox_x2 DECIMAL(10,2) NOT NULL DEFAULT 0,
    bbox_y2 DECIMAL(10,2) NOT NULL DEFAULT 0,
    layout_detection_time DECIMAL(10,3) NOT NULL DEFAULT 0,
    ocr_processing_time DECIMAL(10,3) NOT NULL DEFAULT 0,
    content_analysis_time DECIMAL(10,3) NOT NULL DEFAULT 0,
    total_processing_time DECIMAL(10,3) NOT NULL DEFAULT 0,
    text_length INTEGER NOT NULL DEFAULT 0,
    has_hallucination BOOLEAN NOT NULL DEFAULT FALSE,
    image_path TEXT NOT NULL DEFAULT 'N/A'
);

-- Indexes for common queries
CREATE INDEX idx_line_items_statement ON royalty_line_items(statement_id);
CREATE INDEX idx_line_items_name ON royalty_line_items(item_name);
CREATE INDEX idx_line_items_code ON royalty_line_items(item_code);
CREATE INDEX idx_line_items_channel ON royalty_line_items(channel);
CREATE INDEX idx_summaries_statement ON statement_summaries(statement_id);
CREATE INDEX idx_summaries_category ON statement_summaries(category);
CREATE INDEX idx_metadata_statement ON extraction_metadata(statement_id);
'''
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = TableDataExtractor()
        self.normalizer = ValueNormalizer()
        
        self.statements = []
        self.line_items = []
        self.summaries = []
        self.metadata = []
    
    def process_json_file(self, json_path: str) -> str:
        """Process a single JSON results file and return statement_id."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        statement_id = str(uuid.uuid4())
        pdf_name = data.get('pdf_name', Path(json_path).stem)
        
        # Extract statement info from first page with data
        vendor_name = 'N/A'
        contract_name = 'N/A'
        vendor_number = 'N/A'
        contract_number = 'N/A'
        period_start = 'N/A'
        period_end = 'N/A'
        
        for page in data.get('pages', []):
            structured = page.get('structured_content', {})
            stmt_info = structured.get('statement_info', {})
            if stmt_info:
                if vendor_name == 'N/A' and stmt_info.get('vendor_name') and stmt_info.get('vendor_name') != 'N/A':
                    vendor_name = self.normalizer.clean_string(stmt_info.get('vendor_name')) or 'N/A'
                if contract_name == 'N/A' and stmt_info.get('contract_name') and stmt_info.get('contract_name') != 'N/A':
                    contract_name = self.normalizer.clean_string(stmt_info.get('contract_name')) or 'N/A'
                if vendor_number == 'N/A' and stmt_info.get('vendor_number') and stmt_info.get('vendor_number') != 'N/A':
                    vendor_number = self.normalizer.clean_string(stmt_info.get('vendor_number')) or 'N/A'
                if contract_number == 'N/A' and stmt_info.get('contract_number') and stmt_info.get('contract_number') != 'N/A':
                    contract_number = self.normalizer.clean_string(stmt_info.get('contract_number')) or 'N/A'
                if period_start == 'N/A' and stmt_info.get('period_start') and stmt_info.get('period_start') != 'N/A':
                    period_start = self.normalizer.clean_string(stmt_info.get('period_start')) or 'N/A'
                if period_end == 'N/A' and stmt_info.get('period_end') and stmt_info.get('period_end') != 'N/A':
                    period_end = self.normalizer.clean_string(stmt_info.get('period_end')) or 'N/A'
        
        # Fallback: extract from PDF name
        if vendor_name == 'N/A' and ' - ' in pdf_name:
            parts = pdf_name.split(' - ')
            vendor_name = parts[0].strip()
            if len(parts) > 1:
                contract_name = parts[1].strip()
        
        self.statements.append({
            'id': statement_id,
            'pdf_name': pdf_name,
            'vendor_name': vendor_name,
            'contract_name': contract_name,
            'vendor_number': vendor_number,
            'contract_number': contract_number,
            'period_start': period_start,
            'period_end': period_end,
            'total_pages': data.get('total_pages', 0) or 0,
            'processed_at': data.get('processed_at', datetime.now().isoformat()),
        })
        
        for page in data.get('pages', []):
            self._process_page(statement_id, page)
        
        return statement_id
    
    def _process_page(self, statement_id: str, page: Dict):
        """Process a single page from the results - NO NULLs allowed."""
        page_number = page.get('page_number', 0) or 0
        structured = page.get('structured_content', {})
        proc_meta = page.get('processing_metadata', {})
        
        # Process summary_items (earnings summaries)
        summary_items = structured.get('summary_items', [])
        if isinstance(summary_items, list):
            for item in summary_items:
                if isinstance(item, dict):
                    category = self.normalizer.clean_string(item.get('category')) or 'N/A'
                    subcategory = self.normalizer.clean_string(item.get('subcategory')) or 'N/A'
                    amount = self.normalizer.parse_currency(item.get('amount'))
                    
                    self.summaries.append({
                        'id': str(uuid.uuid4()),
                        'statement_id': statement_id,
                        'page_number': page_number,
                        'category': category,
                        'subcategory': subcategory,
                        'amount': amount if amount is not None else 0,
                    })
        
        # Process line_items (transaction-level detail)
        line_items = structured.get('line_items', [])
        if isinstance(line_items, list):
            for item in line_items:
                if not isinstance(item, dict):
                    continue
                
                # Skip rows with nested structures
                if any(isinstance(v, (dict, list)) for v in item.values() if v is not None):
                    continue
                
                # Extract with NO NULLs - use defaults
                item_name = self.normalizer.clean_string(item.get('item_name') or item.get('item_description')) or 'N/A'
                item_code = self.normalizer.clean_string(item.get('item_code')) or 'N/A'
                channel = self.normalizer.clean_string(item.get('channel')) or 'N/A'
                units = self.normalizer.parse_integer(item.get('units'))
                unit_rate = self.normalizer.parse_currency(item.get('unit_rate') or item.get('unit_price'))
                gross_amount = self.normalizer.parse_currency(item.get('gross_amount'))
                royalty_rate = self.normalizer.parse_currency(item.get('royalty_rate'))
                royalty_amount = self.normalizer.parse_currency(item.get('royalty_amount'))
                
                # Skip rows that are just "N/A" with no data
                if item_name == 'N/A' and (royalty_amount is None or royalty_amount == 0):
                    continue
                
                self.line_items.append({
                    'id': str(uuid.uuid4()),
                    'statement_id': statement_id,
                    'page_number': page_number,
                    'item_name': item_name,
                    'item_code': item_code,
                    'channel': channel,
                    'units': units if units is not None else 0,
                    'unit_rate': unit_rate if unit_rate is not None else 0,
                    'gross_amount': gross_amount if gross_amount is not None else 0,
                    'royalty_rate': royalty_rate if royalty_rate is not None else 0,
                    'royalty_amount': royalty_amount if royalty_amount is not None else 0,
                })
        
        # Process extraction metadata - NO NULLs
        for region in page.get('detected_regions', []):
            bbox = region.get('bbox', [0, 0, 0, 0])
            if len(bbox) < 4:
                bbox = bbox + [0] * (4 - len(bbox))
            
            self.metadata.append({
                'id': str(uuid.uuid4()),
                'statement_id': statement_id,
                'page_number': page_number,
                'region_type': region.get('region_type') or 'unknown',
                'region_confidence': region.get('confidence') or 0,
                'bbox_x1': bbox[0] if bbox[0] is not None else 0,
                'bbox_y1': bbox[1] if bbox[1] is not None else 0,
                'bbox_x2': bbox[2] if bbox[2] is not None else 0,
                'bbox_y2': bbox[3] if bbox[3] is not None else 0,
                'layout_detection_time': proc_meta.get('layout_detection_time') or 0,
                'ocr_processing_time': proc_meta.get('ocr_processing_time') or 0,
                'content_analysis_time': proc_meta.get('content_analysis_time') or 0,
                'total_processing_time': page.get('total_processing_time') or 0,
                'text_length': len(region.get('content', '') or ''),
                'has_hallucination': page.get('has_hallucination', False) or False,
                'image_path': page.get('image_path') or 'N/A',
            })
    
    def export_schema(self) -> str:
        """Export PostgreSQL schema to file."""
        schema_path = self.output_dir / 'schema.sql'
        with open(schema_path, 'w', encoding='utf-8') as f:
            f.write(self.SCHEMA_SQL)
        return str(schema_path)
    
    def export_csv(self) -> Dict[str, str]:
        """Export all tables to CSV files - NO NULLs."""
        csv_files = {}
        
        tables = [
            ('statements', self.statements, [
                'id', 'pdf_name', 'vendor_name', 'contract_name', 
                'vendor_number', 'contract_number', 'period_start', 'period_end',
                'total_pages', 'processed_at'
            ]),
            ('statement_summaries', self.summaries, [
                'id', 'statement_id', 'page_number', 'category',
                'subcategory', 'amount'
            ]),
            ('royalty_line_items', self.line_items, [
                'id', 'statement_id', 'page_number', 'item_name',
                'item_code', 'channel', 'units', 'unit_rate',
                'gross_amount', 'royalty_rate', 'royalty_amount'
            ]),
            ('extraction_metadata', self.metadata, [
                'id', 'statement_id', 'page_number', 'region_type',
                'region_confidence', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                'layout_detection_time', 'ocr_processing_time', 
                'content_analysis_time', 'total_processing_time',
                'text_length', 'has_hallucination', 'image_path'
            ]),
        ]
        
        for table_name, rows, columns in tables:
            csv_path = self.output_dir / f'{table_name}.csv'
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            csv_files[table_name] = str(csv_path)
        
        return csv_files
    
    def export_copy_commands(self) -> str:
        """Generate PostgreSQL COPY commands for importing CSVs."""
        copy_sql = "-- PostgreSQL COPY commands for importing data\n"
        copy_sql += "-- Run these after creating the schema\n\n"
        
        table_columns = {
            'statements': '(id, pdf_name, vendor_name, contract_name, vendor_number, contract_number, period_start, period_end, total_pages, processed_at)',
            'statement_summaries': '(id, statement_id, page_number, category, subcategory, amount)',
            'royalty_line_items': '(id, statement_id, page_number, item_name, item_code, channel, units, unit_rate, gross_amount, royalty_rate, royalty_amount)',
            'extraction_metadata': '(id, statement_id, page_number, region_type, region_confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, layout_detection_time, ocr_processing_time, content_analysis_time, total_processing_time, text_length, has_hallucination, image_path)',
        }
        
        for table, columns in table_columns.items():
            copy_sql += f"\\copy {table}{columns} FROM '{table}.csv' WITH (FORMAT csv, HEADER true);\n"
        
        copy_path = self.output_dir / 'import_data.sql'
        with open(copy_path, 'w', encoding='utf-8') as f:
            f.write(copy_sql)
        
        return str(copy_path)
    
    def export_summary(self) -> Dict[str, Any]:
        """Return export summary statistics."""
        return {
            'statements': len(self.statements),
            'line_items': len(self.line_items),
            'summaries': len(self.summaries),
            'metadata_records': len(self.metadata),
        }


def main():
    parser = argparse.ArgumentParser(
        description='Export JSON results to PostgreSQL-ready format'
    )
    parser.add_argument(
        '--input', '-i',
        default='music_rights/data/output',
        help='Input directory containing JSON result files'
    )
    parser.add_argument(
        '--output', '-o',
        default='music_rights/data/postgres_export',
        help='Output directory for schema and CSV files'
    )
    parser.add_argument(
        '--file', '-f',
        help='Process a single JSON file instead of directory'
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent.parent
    
    if args.file:
        input_files = [Path(args.file)]
    else:
        input_dir = script_dir / args.input
        input_files = list(input_dir.glob('*_results.json'))
    
    if not input_files:
        print(f"No JSON files found in {args.input}")
        sys.exit(1)
    
    output_dir = script_dir / args.output
    exporter = PostgreSQLExporter(str(output_dir))
    
    print("=" * 60)
    print("PostgreSQL Export Tool")
    print("=" * 60)
    
    print(f"\nProcessing {len(input_files)} JSON file(s)...")
    for json_file in input_files:
        print(f"  - {json_file.name}")
        exporter.process_json_file(str(json_file))
    
    print("\nExporting schema...")
    schema_path = exporter.export_schema()
    print(f"  Schema: {schema_path}")
    
    print("\nExporting CSV files...")
    csv_files = exporter.export_csv()
    for table, path in csv_files.items():
        print(f"  {table}: {path}")
    
    print("\nGenerating import commands...")
    import_path = exporter.export_copy_commands()
    print(f"  Import script: {import_path}")
    
    summary = exporter.export_summary()
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    print(f"  Statements:    {summary['statements']}")
    print(f"  Line Items:    {summary['line_items']}")
    print(f"  Summaries:     {summary['summaries']}")
    print(f"  Metadata:      {summary['metadata_records']}")
    print("=" * 60)
    
    print("\nTo import into PostgreSQL:")
    print(f"  1. psql -d your_database -f {schema_path}")
    print(f"  2. cd {output_dir}")
    print(f"  3. psql -d your_database -f import_data.sql")


if __name__ == '__main__':
    main()
