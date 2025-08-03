#!/usr/bin/env python3
"""
📄 PDF Text Extractor
====================

This script extracts text from the example project PDF so we can analyze what made it successful.
"""

import PyPDF2
import sys

def extract_pdf_text(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            print(f"📄 PDF Analysis: {pdf_path}")
            print(f"📊 Number of pages: {len(pdf_reader.pages)}")
            print("=" * 60)
            
            full_text = ""
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                print(f"\n📄 Page {page_num}:")
                print("-" * 40)
                
                page_text = page.extract_text()
                full_text += f"\n--- PAGE {page_num} ---\n{page_text}\n"
                
                # Show first 500 characters of each page
                preview = page_text[:500].replace('\n', ' ')
                print(preview)
                
                if len(page_text) > 500:
                    print("... (truncated)")
            
            # Save full text to file
            output_file = pdf_path.replace('.pdf', '_extracted.txt')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            print(f"\n✅ Full text saved to: {output_file}")
            return full_text
            
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return None

if __name__ == "__main__":
    pdf_file = "Example project that got a perfect grade.pdf"
    text = extract_pdf_text(pdf_file)
    
    if text:
        print(f"\n📊 Total text length: {len(text)} characters")
        print("🎯 Analysis complete! Check the extracted text file for full content.") 