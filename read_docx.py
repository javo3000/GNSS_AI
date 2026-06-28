import zipfile
import xml.etree.ElementTree as ET
import os

docx_path = "BDS_Multiclass_Reliability_Project_Plan.docx"

if not os.path.exists(docx_path):
    print(f"Error: {docx_path} not found.")
    exit(1)

try:
    with zipfile.ZipFile(docx_path) as doc:
        xml_content = doc.read('word/document.xml')
        root = ET.fromstring(xml_content)
        
        # Word processing namespace
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        paragraphs = []
        # Find all paragraphs
        for p in root.findall('.//w:p', ns):
            p_text = []
            # Find all text runs within this paragraph
            for t in p.findall('.//w:t', ns):
                if t.text:
                    p_text.append(t.text)
            if p_text:
                paragraphs.append(''.join(p_text))
            else:
                paragraphs.append('') # Empty line for spacing
        
        full_text = '\n'.join(paragraphs)
        
        # Save output to a text file
        output_txt = "BDS_Multiclass_Project_Plan_extracted.txt"
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Successfully extracted text to {output_txt}")
        print("--- FIRST 50 LINES ---")
        print('\n'.join(full_text.splitlines()[:50]))
except Exception as e:
    print(f"Error reading docx: {e}")
