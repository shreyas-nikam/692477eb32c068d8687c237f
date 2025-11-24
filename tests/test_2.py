import pytest
from definition_8f8a31b5eac0478b84fbd1dc06eb9391 import simulate_financial_document_understanding

# Expected outputs for valid document types as per notebook specification
expected_invoice_data = {
    "Document Type": "Invoice",
    "Invoice Number": "INV-2024-001",
    "Date": "2024-03-15",
    "Vendor": "Global Financial Solutions Inc.",
    "Customer": "Quantum Advisors LLC",
    "Total Amount": "12,345.67 CHF",
    "Currency": "CHF",
    "Items": ["Consulting Fee", "Data Licensing"],
    "Total Items Value": "11,000.00 CHF",
    "Tax Amount": "1,345.67 CHF"
}

expected_annual_report_data = {
    "Document Type": "Annual Report Excerpt",
    "Company": "Innovate Financial Group",
    "Year": 2023,
    "Revenue (Millions)": "5,678",
    "Net Income (Millions)": "1,234",
    "EPS": "4.50",
    "Key Highlight": "Achieved 15% growth in digital assets portfolio."
}

expected_financial_chart_data = {
    "Document Type": "Financial Chart",
    "Chart Title": "Quarterly Net Profits (Millions USD)",
    "Q1 2023": 250,
    "Q2 2023": 280,
    "Q3 2023": 310,
    "Q4 2023": 300,
    "Trend": "Upward trend with slight dip in Q4."
}

@pytest.mark.parametrize(
    "document_type, document_content_placeholder, expected_output",
    [
        # Test Case 1: Valid "invoice" document type with file path placeholder
        ("invoice", "path/to/invoice_image.png", expected_invoice_data),
        # Test Case 2: Valid "annual_report_excerpt" document type with descriptive text placeholder
        ("annual_report_excerpt", "2023 financial highlights", expected_annual_report_data),
        # Test Case 3: Valid "financial_chart" document type with empty placeholder (edge case for content)
        ("financial_chart", "", expected_financial_chart_data),
        # Test Case 4: Unsupported document type (edge case)
        ("unsupported_type", "arbitrary_content.txt", {"Error": "Unsupported document type for simulation."}),
        # Test Case 5: Case-insensitive "document_type" handling (edge case)
        ("AnNuAl_RePoRt_ExCeRpT", "annual_report.pdf", expected_annual_report_data),
    ]
)
def test_simulate_financial_document_understanding(document_type, document_content_placeholder, expected_output):
    """
    Tests the simulate_financial_document_understanding function for various document types,
    including valid scenarios, unsupported types, case-insensitivity, and different content placeholders.
    """
    result = simulate_financial_document_understanding(document_type, document_content_placeholder)
    assert result == expected_output