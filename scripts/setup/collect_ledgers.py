import os
import json
import hashlib
import time

def main():
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("  Error: PROJECT_ROOT environment variable not set.")
        return
        
    try:
        total_expected = int(os.environ.get("TOTAL_EDGES", 0))
    except ValueError:
        print("  Error: TOTAL_EDGES environment variable is invalid.")
        return

    tpm_root = os.path.join(project_root, "runtime", "tpm_state")
    if not os.path.exists(tpm_root):
        print(f"  Error: {tpm_root} not found.")
        return

    print(f"  Collecting hardware pairs from {total_expected} Edge volumes...")
    
    # Dictionary for the ledger that the Fog nodes use
    unified_ledger = {}
    # Dictionary for human-readable names
    identity_mapping = {}  
    collected_dirs = set()
    timeout_seconds = 120
    start_time = time.time()

    # Deterministic polling loop
    while len(collected_dirs) < total_expected:
        if time.time() - start_time > timeout_seconds:
            print(f"  Timeout! Only found {len(collected_dirs)}/{total_expected} TPM states.")
            break

        for dirname in os.listdir(tpm_root):
            if dirname.startswith("edge_") and dirname not in collected_dirs:
                folder_path = os.path.join(tpm_root, dirname)
                id_file = os.path.join(folder_path, "tpm_id.txt")
                pcr_file = os.path.join(folder_path, "clean_pcr.bin")
                
                if os.path.exists(id_file) and os.path.exists(pcr_file):
                    try:
                        with open(id_file, "r") as f:
                            tpm_id = f.read().strip()
                            
                        with open(pcr_file, "rb") as f:
                            data = f.read()
                            
                        idx = data.find(b'\x00\x20')
                        pcr_value = data[idx+2 : idx+34] if idx != -1 else data[-32:]
                        
                        pcr_hex = hashlib.sha256(pcr_value).hexdigest()
                            
                        # 1. Strict Gatekeeper Ledger (ID -> PCR)
                        unified_ledger[tpm_id] = pcr_hex  
                        
                        # 2. Human-readable mapping (Name -> ID)
                        identity_mapping[dirname] = tpm_id
                        
                        collected_dirs.add(dirname) 
                        print(f"    Extracted {dirname} -> ID: {tpm_id[:16]}...")
                    except Exception as e:
                        pass
                        
        time.sleep(2)

    # Write strict Gatekeeper ledger
    ledger_path = os.path.join(tpm_root, "pcr_ledger.json")
    with open(ledger_path, "w") as f:
        json.dump(unified_ledger, f, indent=4)
        
    # Write human-readable mapping
    mapping_path = os.path.join(tpm_root, "tpm_identities.json")
    with open(mapping_path, "w") as f:
        json.dump(identity_mapping, f, indent=4)
        
    if len(collected_dirs) == total_expected:
        print(f"  Admin Ledger and Identity Map generated at {tpm_root} with {len(collected_dirs)} nodes.")
    else:
        print(f"  Warning: Generation incomplete ({len(collected_dirs)} nodes).")

if __name__ == "__main__":
    main()