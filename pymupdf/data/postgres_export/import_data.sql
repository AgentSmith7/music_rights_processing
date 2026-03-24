-- PostgreSQL COPY commands for importing data
-- Run these after creating the schema

\copy statements(id, pdf_name, vendor_name, contract_name, vendor_number, contract_number, period_start, period_end, total_pages, processed_at) FROM 'statements.csv' WITH (FORMAT csv, HEADER true);
\copy statement_summaries(id, statement_id, page_number, category, subcategory, amount) FROM 'statement_summaries.csv' WITH (FORMAT csv, HEADER true);
\copy royalty_line_items(id, statement_id, page_number, item_name, item_code, channel, units, unit_rate, gross_amount, royalty_rate, royalty_amount) FROM 'royalty_line_items.csv' WITH (FORMAT csv, HEADER true);
\copy extraction_metadata(id, statement_id, page_number, region_type, region_confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, layout_detection_time, ocr_processing_time, content_analysis_time, total_processing_time, text_length, has_hallucination, image_path) FROM 'extraction_metadata.csv' WITH (FORMAT csv, HEADER true);
