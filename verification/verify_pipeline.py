import os
import json
import glob
import time
import argparse
import re
from collections import defaultdict

LOG_DIR = "logs/nodes"

def load_logs():
    """
    Fetches raw execution records generating chronological mapping sets corresponding to network iterations.
    """
    all_logs = []
    
    # UPDATED: Use recursive globbing to search all nested container sub-directories
    log_files = glob.glob(f"{LOG_DIR}/**/*.jsonl", recursive=True)
    
    if not log_files:
        print(f"❌ Error: No .jsonl files found in {LOG_DIR}/ or its sub-directories")
        return []

    for file_path in log_files:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        all_logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                        
    # Applies basic sorting constraints assuring relative timebound structure integrity.
    return sorted(all_logs, key=lambda x: x.get('timestamp', ''))

def discover_topology(logs):
    """
    Examines input tracking records deducing total environmental configuration structure mapping exact logical tiers.
    """
    topology = {
        "cloud": set(),
        "fog_clients": set(),
        "fog_servers": set(),
        "edges": set()
    }
    
    for log in logs:
        node = log.get("node", "")
        if "CLOUD" in node:
            topology["cloud"].add(node)
        elif "FOG" in node and "CLIENT" in node:
            topology["fog_clients"].add(node)
        elif "FOG" in node and "SERVER" in node:
            topology["fog_servers"].add(node)
        elif "EDGE" in node:
            topology["edges"].add(node)
            
    return topology

def verify_pipeline():
    """
    Traverses defined functional rounds evaluating completion requirements associated with absolute operational states.
    Returns status variables mapping total iteration continuity success limits.
    """
    # Preserves diagnostic execution results bridging sub-evaluation checks.
    is_pipeline_success = True 
    
    print("\n🔍 Starting Dynamic Pipeline Audit...")
    logs = load_logs()
    if not logs:
        return False

    topology = discover_topology(logs)
    
    print("\n📊 Discovered Network Topology:")
    print(f"  - Cloud Servers: {len(topology['cloud'])}")
    print(f"  - Fog Servers:   {len(topology['fog_servers'])}")
    print(f"  - Fog Clients:   {len(topology['fog_clients'])}")
    print(f"  - Edge Nodes:    {len(topology['edges'])}")
    
    if not all([topology['cloud'], topology['fog_servers'], topology['fog_clients'], topology['edges']]):
        print("\n❌ CRITICAL: Incomplete topology discovered. Missing a node tier entirely.")
        return False

    # Isolates round markers excluding initialization procedures defining independent analysis subsets.
    rounds_data = defaultdict(list)
    for log in logs:
        r = log.get("round")
        
        # Fallback: Extract round from the new hyper-verbose message strings if the JSON 'round' key is null
        if not r:
            msg = log.get("message", "")
            match = re.search(r"for round (\d+)", msg)
            if match:
                r = int(match.group(1))
                
        if r is not None and r > 0: 
            rounds_data[r].append(log)
            
    if not rounds_data:
        print("\n❌ CRITICAL: No round data found. Did the training start?")
        return False

    total_rounds = max(rounds_data.keys())
    print(f"\n⚙️ Analyzing {total_rounds} Training Rounds...\n")

    for current_round in range(1, total_rounds + 1):
        print(f"--- Round {current_round} ---")
        round_logs = rounds_data[current_round]
        
        # Maintains progression logs detailing action execution boundaries.
        actions = defaultdict(set)
        
        for log in round_logs:
            msg = log.get("message", "")
            node = log.get("node", "")
            
            # Verifies initial cloud broadcasting sequence starts.
            if "CLOUD" in node and "Shouting to all FOG clients" in msg:
                actions["cloud_shout"].add(node)
                
            # Validates intermediate client transmission signals (Updated for FogBridge).
            elif "FOG" in node and "CLIENT" in node and "[IPC CLIENT] Connected! Sending START signal" in msg:
                actions["fog_client_start"].add(node)
                
            # Checks operational awakening parameters across intermediate servers (Updated for FogBridge).
            elif "FOG" in node and "SERVER" in node and "[IPC SERVER] START received" in msg:
                actions["fog_server_start"].add(node)
                
            # Identifies successfully terminated individual optimization epochs.
            elif "EDGE" in node and "Epoch" in msg and "complete" in msg:
                actions["edge_train"].add(node)
                
            # Interprets logic branches dictating intermediate aggregation logic processing.
            elif "FOG" in node and "SERVER" in node and ("Caching state" in msg or "Rolling back" in msg or "Falling back" in msg or "No trusted results" in msg):
                actions["fog_server_end"].add(node)
                
            # Ensures terminal data transfer limits reach target environments appropriately.
            elif "CLOUD" in node and "Received weights from" in msg and "[FOG" in msg:
                match = re.search(r"(\[FOG \d+ CLIENT\])", msg)
                if match:
                    actions["fog_client_end"].add(match.group(1))
                    
            # Approves fully enclosed network loops tracking central evaluation metrics.
            elif "CLOUD" in node and "Saved metrics and model to" in msg:
                actions["cloud_eval"].add(node)

        # Calculates execution density against expected configuration totals determining strict adherence limits.
        checks = [
            ("Cloud Initiated", len(actions["cloud_shout"]) == len(topology["cloud"])),
            (f"Fog Clients Bridged ({len(actions['fog_client_start'])}/{len(topology['fog_clients'])})", len(actions["fog_client_start"]) == len(topology['fog_clients'])),
            (f"Fog Servers Woke Up ({len(actions['fog_server_start'])}/{len(topology['fog_servers'])})", len(actions["fog_server_start"]) == len(topology['fog_servers'])),
            (f"Edges Trained ({len(actions['edge_train'])}/{len(topology['edges'])})", len(actions["edge_train"]) == len(topology['edges'])),
            (f"Fog Servers Aggregated ({len(actions['fog_server_end'])}/{len(topology['fog_servers'])})", len(actions["fog_server_end"]) == len(topology['fog_servers'])),
            (f"Fog Clients Delivered Weights to Cloud ({len(actions['fog_client_end'])}/{len(topology['fog_clients'])})", len(actions["fog_client_end"]) == len(topology['fog_clients'])),
            ("Saved metrics and model", len(actions["cloud_eval"]) == len(topology["cloud"]))
        ]

        round_passed = True
        for name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {name}")
            if not passed:
                round_passed = False
                is_pipeline_success = False

        if round_passed:
            print(f"  🟢 Round {current_round} verified successfully.\n")
        else:
            print(f"  🔴 Round {current_round} FAILED. Timeline break detected.\n")

    print("=================================================")
    if is_pipeline_success:
        print("🎉 PIPELINE AUDIT PASSED: Perfect mathematical flow.")
    else:
        print("⚠️ PIPELINE AUDIT FAILED: Data dropped or timeline broken.")
    print("=================================================")
    
    return is_pipeline_success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify the Federated Learning pipeline timeline.")
    parser.add_argument("-w", "--watch", action="store_true", help="Run the verification continuously in a loop.")
    args = parser.parse_args()

    if args.watch:
        print("👀 Entering continuous monitoring mode. Press Ctrl+C to exit.")
        time.sleep(2) 
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"⏱️ Last Refreshed: {time.strftime('%Y-%m-%d %H:%M:%S')} | Press Ctrl+C to exit")
                verify_pipeline()
                time.sleep(1) 
        except KeyboardInterrupt:
            print("\n🛑 Ctrl+C detected. Exiting continuous monitoring mode.")
    else:
        verify_pipeline()