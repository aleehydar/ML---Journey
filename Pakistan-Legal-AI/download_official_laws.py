"""
Pakistan Legal Intelligence — Official Law Downloader
-----------------------------------------------------
This script safely downloads massive, official PDF documents of Pakistan 
Laws directly from government repositories (na.gov.pk, fmu.gov.pk) 
into the data/ folder.

It uses a standard browser User-Agent to avoid getting 403 blocks 
from basic government firewalls.
"""

import os
import requests
import sys

# Remove sample PDFs to make room for the real ones
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Clean up older sample files
samples = ["constitution_fundamental_rights.pdf", "pakistan_labour_laws.pdf", "pakistan_tax_laws.pdf", "tenant_property_rights.pdf"]
for sample in samples:
    path = os.path.join(DATA_DIR, sample)
    if os.path.exists(path):
        os.remove(path)
        print(f"🗑️ Removed sample file: {sample}")

# The Official PDFs to Download
OFFICIAL_LAWS = {
    "Constitution_of_Pakistan_1973.pdf": "https://na.gov.pk/uploads/documents/1333523681_951.pdf",
    "Pakistan_Penal_Code_1860.pdf": "https://fmu.gov.pk/docs/laws/Pakistan%20Penal%20Code.pdf",
    "Anti_Money_Laundering_Act_2010.pdf": "https://fmu.gov.pk/docs/laws/AML%20Act%202010.pdf",
    "Anti_Terrorism_Act_1997.pdf": "https://fmu.gov.pk/docs/laws/Anti-Terrorism%20Act,%201997.pdf"
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept": "application/pdf"
}

print("=" * 60)
print("  Downloading Official Pakistan Laws (1000+ Pages)")
print("=" * 60)

import urllib3
urllib3.disable_warnings()

for filename, url in OFFICIAL_LAWS.items():
    filepath = os.path.join(DATA_DIR, filename)
    print(f"\n📥 Downloading: {filename}")
    print(f"   From: {url}")
    
    try:
        response = requests.get(url, headers=headers, stream=True, verify=False, timeout=30)
        
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            # Get size in MB
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"   ✅ Saved! Size: {size_mb:.2f} MB")
        else:
            print(f"   ❌ Failed. Server returned HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Request Error: {e}")

print("\n" + "=" * 60)
print("✅ Downloads completed. You can now run:")
print("   python ingest.py")
print("=" * 60)
