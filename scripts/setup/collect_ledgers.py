import os
import json
import base64
import hashlib

def main():
    # Fetch the root directory dynamically passed from the bash script
    project_root = os.environ["PROJECT_ROOT"]
    if not project_root:
        print("🛑 Error: PROJECT_ROOT environment variable not set.")
        return

    tpm_root = os.path.join(project_root, "runtime", "tpm_state")
    unified_ledger = {}

    if os.path.exists(tpm_root):
        print("🗂️ Collecting hardware pairs from Edge volumes...")
        for dirname in os.listdir(tpm_root):
            if dirname.startswith("edge_"):
                folder_path = os.path.join(tpm_root, dirname)
                id_file = os.path.join(folder_path, "tpm_id.txt")
                pcr_file = os.path.join(folder_path, "clean_pcr.bin")
                
                if os.path.exists(id_file) and os.path.exists(pcr_file):
                    with open(id_file, "r") as f:
                        tpm_id = f.read().strip()
                        
                    with open(pcr_file, "rb") as f:
                        data = f.read()
                        
                    # Locate the 32-byte SHA-256 PCR value inside the binary structure
                    idx = data.find(b'\x00\x20')
                    pcr_value = data[idx+2 : idx+34] if idx != -1 else data[-32:]
                    
                    # Compute the hardware pcrDigest expected by the fog server
                    pcr_hex = hashlib.sha256(pcr_value).hexdigest()
                        
                    unified_ledger[tpm_id] = pcr_hex
                    print(f"  ✅ Extracted {dirname} -> ID: {tpm_id[:16]}...")
                else:
                    print(f"  ⚠️ Warning: {dirname} is missing files. Wait for container to finish booting.")

        ledger_path = os.path.join(tpm_root, "pcr_ledger.json")
        with open(ledger_path, "w") as f:
            json.dump(unified_ledger, f, indent=4)
        print(f"💾 Unified Admin Ledger generated at {ledger_path} with {len(unified_ledger)} nodes secured.")
    else:
        print(f"🛑 Error: {tpm_root} not found.")

if __name__ == "__main__":
    main()