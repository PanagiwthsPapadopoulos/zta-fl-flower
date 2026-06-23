import os
import json
import base64

def main():
    # Fetch the root directory dynamically passed from the bash script
    project_root = os.environ.get("PROJECT_ROOT")
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
                        pcr_b64 = base64.b64encode(f.read()).decode('utf-8')
                        
                    unified_ledger[tpm_id] = pcr_b64
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