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
    
    unified_ledger = {}
    timeout_seconds = 120  # 2 minute failsafe timeout
    start_time = time.time()

    # Deterministic polling loop
    while len(unified_ledger) < total_expected:
        if time.time() - start_time > timeout_seconds:
            print(f"  Timeout! Only found {len(unified_ledger)}/{total_expected} TPM states.")
            break

        for dirname in os.listdir(tpm_root):
            if dirname.startswith("edge_") and dirname not in unified_ledger:
                folder_path = os.path.join(tpm_root, dirname)
                id_file = os.path.join(folder_path, "tpm_id.txt")
                pcr_file = os.path.join(folder_path, "clean_pcr.bin")
                
                if os.path.exists(id_file) and os.path.exists(pcr_file):
                    try:
                        with open(id_file, "r") as f:
                            tpm_id = f.read().strip()
                            
                        with open(pcr_file, "rb") as f:
                            data = f.read()
                            
                        # Locate the 32-byte SHA-256 PCR value inside the binary structure
                        idx = data.find(b'\x00\x20')
                        pcr_value = data[idx+2 : idx+34] if idx != -1 else data[-32:]
                        
                        # Compute the hardware pcrDigest expected by the fog server
                        pcr_hex = hashlib.sha256(pcr_value).hexdigest()
                            
                        unified_ledger[dirname] = pcr_hex  # Track by dirname to prevent double-counting
                        print(f"    Extracted {dirname} -> ID: {tpm_id[:16]}...")
                    except Exception as e:
                        pass # File might be mid-write; we will catch it on the next loop iteration
                        
        time.sleep(2) # Prevent CPU thrashing while waiting

    ledger_path = os.path.join(tpm_root, "pcr_ledger.json")
    with open(ledger_path, "w") as f:
        json.dump(unified_ledger, f, indent=4)
        
    if len(unified_ledger) == total_expected:
        print(f"  Unified Admin Ledger generated at {ledger_path} with all {len(unified_ledger)} nodes secured.")
    else:
        print(f"  Warning: Unified Admin Ledger generated incompletely ({len(unified_ledger)} nodes).")

if __name__ == "__main__":
    main()