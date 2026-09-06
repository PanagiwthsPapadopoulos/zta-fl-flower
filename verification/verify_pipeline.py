import os
import re
import glob
import json
import time
import argparse
from collections import defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich import box
from rich.progress import Progress, BarColumn, TextColumn

LOG_DIR = "logs/nodes"
console = Console()

def load_logs():
    """Load and sort .jsonl logs."""
    all_logs = []
    log_files = glob.glob(f"{LOG_DIR}/**/*.jsonl", recursive=True)
    if not log_files:
        return []

    for file_path in log_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            all_logs.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception:
            continue

    return sorted(all_logs, key=lambda x: x.get('timestamp', ''))

def discover_topology(logs):
    """Map nodes to topology tiers."""
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

def get_progress_bar(label, completed, total):
    """Create a progress bar."""
    progress = Progress(
        TextColumn("{task.description}", justify="left"),
        BarColumn(bar_width=35, complete_style="green", finished_style="green", style="grey23"),
        TextColumn("[bold white]{task.percentage:>3.0f}%"),
        TextColumn("[dim white]{task.completed}/{task.total}"),
        expand=False,
    )
    
    safe_total = max(1, total)
    padded_label = f"{label:<25}"
    
    progress.add_task(f"[dim white]{padded_label}", total=safe_total, completed=completed)
    return progress

def parse_round_state(round_logs, topology):
    """Parse round state."""
    state = {
        "nodes_booted": set(),
        "cloud_init": False,
        "fog_clients_started": set(),
        "fog_servers_started": set(),
        "edges_trained": set(),
        "edge_epochs": {},           
        "edge_tokens_generated": set(),
        "edge_tokens_verified": set(),
        "edge_tokens_rejected": {},  
        "fog_servers_aggregated": set(),
        "fog_live_status": {},
        "fog_clients_delivered": set(),
        "cloud_evaluated": False,
        "eval_metrics": None,
        "latest_edge_msg": {},
        "latest_fog_msg": {}         
    }

    last_node_error = {}

    for log in round_logs:
        msg = log.get("message", "")
        node = log.get("node", "")
        level = log.get("level", "")

        if node:
            state["nodes_booted"].add(node)
            
        if "FOG" in node:
            state["latest_fog_msg"][node] = msg
            
        if level == "ERROR":
            last_node_error[node] = msg

        if "CLOUD" in node and "Shouting to all FOG clients" in msg:
            state["cloud_init"] = True
        elif "FOG" in node and "CLIENT" in node and "[IPC CLIENT] Connected! Sending START signal" in msg:
            state["fog_clients_started"].add(node)
        elif "FOG" in node and "SERVER" in node and "[IPC SERVER] START received" in msg:
            state["fog_servers_started"].add(node)
        elif "EDGE" in node:
            state["latest_edge_msg"][node] = msg
            
            epoch_match = re.search(r"Epoch\s*(\d+)(?:\s*(?:/|of)\s*(\d+))?", msg, re.IGNORECASE)
            if epoch_match:
                curr = epoch_match.group(1)
                tot = epoch_match.group(2)
                state["edge_epochs"][node] = f"Epoch {curr}/{tot}" if tot else f"Epoch {curr}"
                
            if "Epoch" in msg and "complete" in msg:
                state["edges_trained"].add(node)
            if "[TPM-GENERATE] Final Plaintext JSON Token" in msg or "hardware quote structure" in msg:
                state["edge_tokens_generated"].add(node)
        elif "FOG" in node and "SERVER" in node:
            if match := re.search(r"Ingesting Cryptographic Result for (\[EDGE [^\]]+\])", msg):
                state["edge_tokens_verified"].add(match.group(1))
            if match := re.search(r"REJECTED: Attestation/PCR mismatch for (\[EDGE [^\]]+\])", msg):
                edge_id = match.group(1)
                state["edge_tokens_rejected"][edge_id] = last_node_error.get(node, "Unknown Error")
            if any(k in msg for k in ["Relaying", "No trusted results", "Bypassing"]):
                clean_msg = msg.replace("[IPC SERVER] ", "").replace(f"{node} ", "")
                state["fog_live_status"][node] = clean_msg
                
            if any(k in msg for k in ["Saved 3-pillar Fog state", "Relaying", "No trusted results", "Caching state", "Falling back"]):
                state["fog_servers_aggregated"].add(node)
        elif "CLOUD" in node and "Received weights from" in msg and "[FOG" in msg:
            if match := re.search(r"(\[FOG \d+ CLIENT\])", msg):
                state["fog_clients_delivered"].add(match.group(1))
        elif "CLOUD" in node:
            if match := re.search(r"CLOUD Eval Round \d+ \| Loss: ([\d.]+) \| Acc: ([\d.]+) \| F1: ([\d.]+)", msg):
                state["eval_metrics"] = {"loss": match.group(1), "acc": match.group(2), "f1": match.group(3)}
            if "Saved metrics and model" in msg:
                state["cloud_evaluated"] = True

    if state["cloud_evaluated"] and not state["fog_clients_delivered"]:
        state["fog_clients_delivered"] = set(topology["fog_clients"])

    return state

def build_round_panel(r_num, max_rounds, state, topology, is_current_round=True):
    """Build the TUI panel."""
    n_clouds = max(1, len(topology["cloud"]))
    n_fog_c = max(1, len(topology["fog_clients"]))
    n_fog_s = max(1, len(topology["fog_servers"]))
    n_edges = max(1, len(topology["edges"]))
    
    cloud_count = n_clouds if state["cloud_init"] else min(sum(1 for n in state["nodes_booted"] if "CLOUD" in n), n_clouds)
    tot_fog = n_fog_s + n_fog_c
    fog_count = min(sum(1 for n in state["nodes_booted"] if "FOG" in n), tot_fog)
    edge_count = min(sum(1 for n in state["nodes_booted"] if "EDGE" in n), n_edges)

    p1_complete = (cloud_count == n_clouds) and (fog_count == tot_fog) and (edge_count == n_edges)
    p2_complete = len(state["fog_clients_started"]) == n_fog_c and len(state["fog_servers_started"]) == n_fog_s
    p3_complete = len(state["edges_trained"]) == n_edges
    p4_complete = len(state["edge_tokens_generated"]) == n_edges
    
    total_processed_tokens = len(state["edge_tokens_verified"]) + len(state["edge_tokens_rejected"])
    p5_complete = total_processed_tokens == n_edges
    p6_complete = len(state["fog_servers_aggregated"]) == n_fog_s
    p7_complete = state["cloud_evaluated"]

    t = Table.grid(padding=(0, 0))
    has_pending_phase = False
    
    def get_completed_title(text):
        return f"[bold green]✔[/] [bold white]{text}[/]" if is_current_round else f"[dim green]✔[/] [dim white]{text}[/]"

    # ---------------------------
    # PHASE 1: Initialization
    # ---------------------------
    if p1_complete:
        t.add_row(get_completed_title("Initialization Completed"))
        if is_current_round:
            t.add_row(get_progress_bar("  Cloud Nodes:", n_clouds, n_clouds))
            t.add_row(get_progress_bar("  Fog Nodes:", tot_fog, tot_fog))
            t.add_row(get_progress_bar("  Edge Nodes:", n_edges, n_edges))
    elif not has_pending_phase:
        t.add_row("[bold white]Executing Fog Aggregation....[/]")
        t.add_row(get_progress_bar("  Fog Aggregation:", len(state["fog_servers_aggregated"]), n_fog_s))
        t.add_row("  [dim white]Awaiting for:[/]")
        for f_serv in sorted(topology["fog_servers"]):
            if f_serv not in state["fog_servers_aggregated"]:
                curr_status = state.get("fog_live_status", {}).get(f_serv, "Aggregating...")
                short_status = (curr_status[:60] + "...") if len(curr_status) > 60 else curr_status
                t.add_row(f"  [bold white]{f_serv}:[/] [green]{short_status}[/]")
        has_pending_phase = True

    # ---------------------------
    # PHASE 2: Bridging
    # ---------------------------
    if p2_complete:
        t.add_row(get_completed_title("Bridging Completed"))
        if is_current_round:
            t.add_row(get_progress_bar("  Fog Nodes Bridged:", n_fog_c, n_fog_c))
    elif has_pending_phase and is_current_round:
        t.add_row("[dim green]▶ Upcoming: Bridging Nodes...[/]")
    elif not has_pending_phase:
        t.add_row("[bold white]Awaiting Bridging...[/]")
        bridged_count = min(len(state["fog_clients_started"]), len(state["fog_servers_started"]))
        t.add_row(get_progress_bar("  Fog Nodes Bridged:", bridged_count, n_fog_c))
        
        t.add_row("  [bold white]Awaiting Fog Servers:[/]")
        for f_serv in sorted(topology["fog_servers"]):
            if f_serv not in state["fog_servers_started"]:
                msg = state["latest_fog_msg"].get(f_serv, "Awaiting initial log message...")
                short_msg = (msg[:60] + "...") if len(msg) > 60 else msg
                color = "bold green" if any(k in msg for k in ["Booting", "Listening", "Blocking", "established", "START"]) else "bold red"
                t.add_row(f"    [{color}]{f_serv}:[/] [dim white]{short_msg}[/]")
                
        t.add_row("  [bold white]Awaiting Fog Clients:[/]")
        for f_client in sorted(topology["fog_clients"]):
            if f_client not in state["fog_clients_started"]:
                msg = state["latest_fog_msg"].get(f_client, "Awaiting initial log message...")
                short_msg = (msg[:60] + "...") if len(msg) > 60 else msg
                color = "bold green" if any(k in msg for k in ["Attempting", "established", "Connected", "dispatched"]) else "bold red"
                t.add_row(f"    [{color}]{f_client}:[/] [dim white]{short_msg}[/]")
                
        has_pending_phase = True

    # ---------------------------
    # PHASE 3: Training
    # ---------------------------
    if p3_complete:
        t.add_row(get_completed_title("Edges Trained"))
        if is_current_round:
            t.add_row(get_progress_bar("  Progress:", n_edges, n_edges))
    elif has_pending_phase and is_current_round:
        t.add_row("[dim green]▶ Upcoming: Executing Training...[/]")
    elif not has_pending_phase:
        t.add_row("[bold white]Executing Training[/]")
        t.add_row(get_progress_bar("  Progress:", len(state["edges_trained"]), n_edges))
        t.add_row("  [dim white]Awaiting for:[/]")
        for edge in sorted(topology["edges"]):
            if edge not in state["edges_trained"]:
                if edge in state.get("edge_epochs", {}):
                    t.add_row(f"  [bold white]{edge}:[/] [green]Training: {state['edge_epochs'][edge]}[/]")
                else:
                    msg = state["latest_edge_msg"].get(edge, "Awaiting initial log message...")
                    short_msg = (msg[:60] + "...") if len(msg) > 60 else msg
                    t.add_row(f"  [bold white]{edge}[/] [dim white]latest log message:[/] [dim white]{short_msg}[/]")
        has_pending_phase = True

    # ---------------------------
    # PHASE 4: Token Generation
    # ---------------------------
    if p4_complete:
        t.add_row(get_completed_title("Tokens Generated"))
        if is_current_round:
            t.add_row(get_progress_bar("  Progress:", n_edges, n_edges))
    elif has_pending_phase and is_current_round:
        t.add_row("[dim green]▶ Upcoming: Generating Attestation Tokens...[/]")
    elif not has_pending_phase:
        t.add_row("[bold white]Generating Attestation Tokens[/]")
        t.add_row(get_progress_bar("  Progress:", len(state["edge_tokens_generated"]), n_edges))
        t.add_row("  [dim white]Awaiting for:[/]")
        for edge in sorted(topology["edges"]):
            if edge not in state["edge_tokens_generated"]:
                msg = state["latest_edge_msg"].get(edge, "Awaiting token...")
                short_msg = (msg[:60] + "...") if len(msg) > 60 else msg
                t.add_row(f"  [bold white]{edge}[/] [dim white]latest log message:[/] [dim white]{short_msg}[/]")
        has_pending_phase = True

    # ---------------------------
    # PHASE 5: Verification 
    # ---------------------------
    if p5_complete:
        t.add_row(get_completed_title("Attestation Tokens Verification Completed"))
        if is_current_round:
            t.add_row(get_progress_bar("  Tokens received:", n_edges, n_edges))
            t.add_row(get_progress_bar("  Tokens parsed:", n_edges, n_edges))
            
        if len(state["edge_tokens_rejected"]) > 0:
            for rej, error_msg in state["edge_tokens_rejected"].items():
                t.add_row(f"  [bold red]✖ Rejected: {rej} - {error_msg}[/]")
                
    elif has_pending_phase and is_current_round:
        t.add_row("[dim green]▶ Upcoming: Verifying Attestation Tokens...[/]")
    elif not has_pending_phase:
        t.add_row("[bold white]Verifying Attestation Tokens....[/]")
        
        tokens_received = len(state["edge_tokens_generated"])
        t.add_row(get_progress_bar("  Tokens received:", tokens_received, n_edges))
        t.add_row(get_progress_bar("  Tokens parsed:", total_processed_tokens, n_edges))
        
        if len(state["edge_tokens_rejected"]) > 0:
            for rej, error_msg in state["edge_tokens_rejected"].items():
                t.add_row(f"  [bold red]✖ Rejected: {rej} - {error_msg}[/]")
        has_pending_phase = True

    # ---------------------------
    # PHASE 6: Fog Aggregation
    # ---------------------------
    if p6_complete:
        t.add_row(get_completed_title("Fog Aggregation Completed"))
        if is_current_round:
            t.add_row(get_progress_bar("  Fog Aggregation:", n_fog_s, n_fog_s))
    elif has_pending_phase and is_current_round:
        t.add_row("[dim green]▶ Upcoming: Executing Fog Aggregation...[/]")
    elif not has_pending_phase:
        t.add_row("[bold white]Executing Fog Aggregation....[/]")
        t.add_row(get_progress_bar("  Fog Aggregation:", len(state["fog_servers_aggregated"]), n_fog_s))
        has_pending_phase = True

    # ---------------------------
    # PHASE 7: Global Evaluation
    # ---------------------------
    calc_done = bool(state.get("eval_metrics"))
    save_done = state.get("cloud_evaluated")

    calc_finished = calc_done or not is_current_round
    save_finished = save_done or not is_current_round

    calc_status = "[bold green]✔[/]" if calc_done else ("[bold green]◯[/]" if is_current_round else "[bold red]✖[/]")
    save_status = "[bold green]✔[/]" if save_done else ("[bold green]◯[/]" if is_current_round else "[bold red]✖[/]")

    calc_text = "Calculated Metrics" if calc_finished else "Calculating Metrics..."
    save_text = "Saved Model & Metrics" if save_finished else "Saving Model & Metrics..."

    if p7_complete:
        t.add_row(get_completed_title("Global Model Evaluation Completed"))
        if is_current_round:
            t.add_row(get_progress_bar("  Received updates:", n_fog_c, n_fog_c))
        
        t.add_row(f"  {calc_status} [dim white]{calc_text}[/]")
        t.add_row(f"  {save_status} [dim white]{save_text}[/]")
        
    elif has_pending_phase and is_current_round:
        t.add_row("[dim green]▶ Upcoming: Completing Evaluation...[/]")
    elif not has_pending_phase:
        t.add_row("[bold white]Evaluating Global Model....[/]")
        t.add_row(get_progress_bar("  Received updates:", len(state["fog_clients_delivered"]), n_fog_c))
        
        t.add_row(f"  {calc_status} [dim white]{calc_text}[/]")
        t.add_row(f"  {save_status} [dim white]{save_text}[/]")
        
        has_pending_phase = True

    style = "dim" if p7_complete and not is_current_round else ""
    title_color = "bold white" if is_current_round else "dim grey50"
    
    return Panel(t, title=f"[{title_color}]Round {r_num}/{max_rounds}[/]", title_align="left", border_style="grey37", style=style, box=box.SQUARE)

def generate_dashboard(logs):
    """Build TUI dashboard."""
    if not logs:
        return Panel("[dim white]No log files discovered.[/]", box=box.SQUARE)

    topology = discover_topology(logs)
    rounds_data = defaultdict(list)

    for log in logs:
        r = log.get("round")
        msg = log.get("message", "")
        
        if not r:
            if match := re.search(r"(?:for round|CLOUD Eval Round) (\d+)", msg):
                r = int(match.group(1))
                
        if r is not None and r > 0:
            rounds_data[r].append(log)

    if not rounds_data:
        return Panel("[dim white]Awaiting federated round initialization...[/]", box=box.SQUARE)

    max_rounds = max(rounds_data.keys())
    main_table = Table.grid(expand=True)

    pipeline_completed = False

    for r_num in range(1, max_rounds + 1):
        round_logs = rounds_data[r_num]
        state = parse_round_state(round_logs, topology)
        is_latest = (r_num == max_rounds)
        
        main_table.add_row(build_round_panel(r_num, max_rounds, state, topology, is_current_round=is_latest))
        
        if is_latest and state["cloud_evaluated"]:
            pipeline_completed = True

    if pipeline_completed:
        main_table.add_row("")
        main_table.add_row(Panel("[bold white]✦ Pipeline Execution Completed Successfully ✦[/]", border_style="white", box=box.SQUARE))

    return main_table

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--watch", action="store_true", help="Run in continuous live watch mode")
    args = parser.parse_args()

    if args.watch:
        with Live(console=console, screen=True, auto_refresh=False) as live:
            try:
                while True:
                    logs = load_logs()
                    live.update(generate_dashboard(logs), refresh=True)
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    else:
        logs = load_logs()
        console.print(generate_dashboard(logs))

if __name__ == "__main__":
    main()