"""Pipeline monitor: unchanged phase logic, scroll-safe terminal output.

Use -w for append-only history; --live for a bounded alternate-screen display.
"""
import os
import re
import glob
import json
import time
import argparse
import math
import signal
import sys
from contextlib import contextmanager
from collections import defaultdict

from rich.console import Console
from rich.markup import escape
from rich.segment import Segment
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
                curr = int(epoch_match.group(1))
                tot = int(epoch_match.group(2)) if epoch_match.group(2) else None
                state["edge_epochs"][node] = f"Epoch {curr}/{tot}" if tot else f"Epoch {curr}"

                # Only mark as fully trained when the FINAL epoch reports "complete",
                # not merely when any epoch reports "complete".
                if tot is not None and curr == tot and "complete" in msg.lower():
                    state["edges_trained"].add(node)
            if "[TPM-GENERATE] Final Plaintext JSON Token" in msg or "hardware quote structure" in msg:
                state["edge_tokens_generated"].add(node)
        elif "FOG" in node and "SERVER" in node:
            # Added .*? to account for hardware IDs preceding the [EDGE X] tag
            if match := re.search(r"Ingesting Cryptographic Result for .*?(\[EDGE [^\]]+\])", msg):
                state["edge_tokens_verified"].add(match.group(1))
            if match := re.search(r"REJECTED: Attestation/PCR mismatch for .*?(\[EDGE [^\]]+\])", msg):
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

def build_round_panel(r_num, max_rounds, state, topology, is_current_round=True, is_expanded=True):
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

    # 1. Determine if the entire round is 100% finished
    is_round_complete = all([p1_complete, p2_complete, p3_complete, p4_complete, p5_complete, p6_complete, p7_complete])

    # 2. Determine the furthest active phase to prevent marking previous incomplete steps as "Upcoming"
    highest_phase = 1
    if p7_complete or state.get("cloud_evaluated") or len(state.get("fog_clients_delivered", [])) > 0 or state.get("eval_metrics") is not None: 
        highest_phase = 7
    elif p6_complete or len(state.get("fog_servers_aggregated", [])) > 0 or len(state.get("fog_live_status", {})) > 0: 
        highest_phase = 6
    elif p5_complete or total_processed_tokens > 0 or len(state.get("edge_tokens_rejected", {})) > 0: 
        highest_phase = 5
    elif p4_complete or len(state.get("edge_tokens_generated", [])) > 0: 
        highest_phase = 4
    elif p3_complete or len(state.get("edges_trained", [])) > 0 or len(state.get("edge_epochs", {})) > 0: 
        highest_phase = 3
    elif p2_complete or len(state.get("fog_clients_started", [])) > 0 or len(state.get("fog_servers_started", [])) > 0: 
        highest_phase = 2

    t = Table.grid(padding=(0, 0))
    has_pending_phase = False
    
    def get_completed_title(text):
        # Changed completed color to green text instead of white text
        return f"[bold green]✔ {text}[/]" if is_current_round else f"[dim green]✔ {text}[/]"

    # ---------------------------
    # PHASE 1: Initialization
    # ---------------------------
    if p1_complete:
        t.add_row(get_completed_title("Initialization Completed"))
        if is_expanded and not is_round_complete:
            t.add_row(get_progress_bar("  Cloud Nodes:", n_clouds, n_clouds))
            t.add_row(get_progress_bar("  Fog Nodes:", tot_fog, tot_fog))
            t.add_row(get_progress_bar("  Edge Nodes:", n_edges, n_edges))
    elif has_pending_phase and highest_phase < 1 and is_expanded:
        t.add_row("[dim white]▶ Upcoming: Initialization...[/]")
    else:
        t.add_row("[bold white]Executing Initialization...[/]")
        t.add_row(get_progress_bar("  Cloud Nodes:", cloud_count, n_clouds))
        t.add_row(get_progress_bar("  Fog Nodes:", fog_count, tot_fog))
        t.add_row(get_progress_bar("  Edge Nodes:", edge_count, n_edges))
        has_pending_phase = True

    # ---------------------------
    # PHASE 2: Bridging
    # ---------------------------
    if p2_complete:
        t.add_row(get_completed_title("Bridging Completed"))
        if is_current_round and not is_round_complete:
            t.add_row(get_progress_bar("  Fog Nodes Bridged:", n_fog_c, n_fog_c))
    elif has_pending_phase and highest_phase < 2 and is_current_round:
        t.add_row("[dim white]▶ Upcoming: Bridging Nodes...[/]")
    else:
        t.add_row("[bold white]Awaiting Bridging...[/]")
        bridged_count = min(len(state["fog_clients_started"]), len(state["fog_servers_started"]))
        t.add_row(get_progress_bar("  Fog Nodes Bridged:", bridged_count, n_fog_c))
        
        t.add_row("  [bold white]Awaiting Fog Servers:[/]")
        for f_serv in sorted(topology["fog_servers"]):
            if f_serv not in state["fog_servers_started"]:
                msg = state["latest_fog_msg"].get(f_serv, "Awaiting initial log message...")
                short_msg = (msg[:60] + "...") if len(msg) > 60 else msg
                color = "bold green" if any(k in msg for k in ["Booting", "Listening", "Blocking", "established", "START"]) else "bold white"
                t.add_row(f"    [{color}]{safe_markup(f_serv)}:[/] [dim white]{safe_markup(short_msg)}[/]")
                
        t.add_row("  [bold white]Awaiting Fog Clients:[/]")
        for f_client in sorted(topology["fog_clients"]):
            if f_client not in state["fog_clients_started"]:
                msg = state["latest_fog_msg"].get(f_client, "Awaiting initial log message...")
                short_msg = (msg[:60] + "...") if len(msg) > 60 else msg
                color = "bold green" if any(k in msg for k in ["Attempting", "established", "Connected", "dispatched"]) else "bold white"
                t.add_row(f"    [{color}]{safe_markup(f_client)}:[/] [dim white]{safe_markup(short_msg)}[/]")
                
        has_pending_phase = True

    # ---------------------------
    # PHASE 3: Training
    # ---------------------------
    if p3_complete:
        t.add_row(get_completed_title("Edges Trained"))
        if is_current_round and not is_round_complete:
            t.add_row(get_progress_bar("  Progress:", n_edges, n_edges))
    elif has_pending_phase and highest_phase < 3 and is_current_round:
        t.add_row("[dim white]▶ Upcoming: Executing Training...[/]")
    else:
        t.add_row("[bold white]Executing Training[/]")
        t.add_row(get_progress_bar("  Progress:", len(state["edges_trained"]), n_edges))
        t.add_row("  [dim white]Awaiting for:[/]")
        for edge in sorted(topology["edges"]):
            if edge not in state["edges_trained"]:
                if edge in state.get("edge_epochs", {}):
                    t.add_row(f"  [bold white]{safe_markup(edge)}:[/] [green]Training: {safe_markup(state['edge_epochs'][edge])}[/]")
                else:
                    msg = state["latest_edge_msg"].get(edge, "Awaiting initial log message...")
                    short_msg = (msg[:60] + "...") if len(msg) > 60 else msg
                    t.add_row(f"  [bold white]{safe_markup(edge)}[/] [dim white]latest log message:[/] [dim white]{safe_markup(short_msg)}[/]")
        has_pending_phase = True

    # ---------------------------
    # PHASE 4: Token Generation
    # ---------------------------
    if p4_complete:
        t.add_row(get_completed_title("Tokens Generated"))
        if is_current_round and not is_round_complete:
            t.add_row(get_progress_bar("  Progress:", n_edges, n_edges))
    elif has_pending_phase and highest_phase < 4 and is_current_round:
        t.add_row("[dim white]▶ Upcoming: Generating Attestation Tokens...[/]")
    else:
        t.add_row("[bold white]Generating Attestation Tokens[/]")
        t.add_row(get_progress_bar("  Progress:", len(state["edge_tokens_generated"]), n_edges))
        t.add_row("  [dim white]Awaiting for:[/]")
        for edge in sorted(topology["edges"]):
            if edge not in state["edge_tokens_generated"]:
                msg = state["latest_edge_msg"].get(edge, "Awaiting token...")
                short_msg = (msg[:60] + "...") if len(msg) > 60 else msg
                t.add_row(f"  [bold white]{safe_markup(edge)}[/] [dim white]latest log message:[/] [dim white]{safe_markup(short_msg)}[/]")
        has_pending_phase = True

    # ---------------------------
    # PHASE 5: Verification 
    # ---------------------------
    if p5_complete:
        t.add_row(get_completed_title("Attestation Tokens Verification Completed"))
        if is_current_round and not is_round_complete:
            t.add_row(get_progress_bar("  Tokens received:", n_edges, n_edges))
            t.add_row(get_progress_bar("  Tokens parsed:", n_edges, n_edges))
            
        if len(state["edge_tokens_rejected"]) > 0:
            for rej, error_msg in state["edge_tokens_rejected"].items():
                t.add_row(f"  [bold red]✖ Rejected: {safe_markup(rej)} - {safe_markup(error_msg)}[/]")
    elif has_pending_phase and highest_phase < 5 and is_current_round:
        t.add_row("[dim white]▶ Upcoming: Verifying Attestation Tokens...[/]")
    else:
        t.add_row("[bold white]Verifying Attestation Tokens....[/]")
        tokens_received = len(state["edge_tokens_generated"])
        t.add_row(get_progress_bar("  Tokens received:", tokens_received, n_edges))
        t.add_row(get_progress_bar("  Tokens parsed:", total_processed_tokens, n_edges))
        
        if len(state["edge_tokens_rejected"]) > 0:
            for rej, error_msg in state["edge_tokens_rejected"].items():
                t.add_row(f"  [bold red]✖ Rejected: {safe_markup(rej)} - {safe_markup(error_msg)}[/]")
        has_pending_phase = True

    # ---------------------------
    # PHASE 6: Fog Aggregation
    # ---------------------------
    if p6_complete:
        t.add_row(get_completed_title("Fog Aggregation Completed"))
        if is_expanded and not is_round_complete:
            t.add_row(get_progress_bar("  Fog Aggregation:", n_fog_s, n_fog_s))
    elif has_pending_phase and highest_phase < 6 and is_expanded:
        t.add_row("[dim white]▶ Upcoming: Executing Fog Aggregation...[/]")
    else:
        t.add_row("[bold white]Executing Fog Aggregation....[/]")
        t.add_row(get_progress_bar("  Fog Aggregation:", len(state["fog_servers_aggregated"]), n_fog_s))
        t.add_row("  [dim white]Awaiting for:[/]")
        for f_serv in sorted(topology["fog_servers"]):
            if f_serv not in state["fog_servers_aggregated"]:
                msg = state["latest_fog_msg"].get(f_serv, "Awaiting initial log message...")
                short_msg = (msg[:60] + "...") if len(msg) > 60 else msg
                t.add_row(f"  [bold white]{safe_markup(f_serv)}[/] [dim white]latest log message:[/] [dim white]{safe_markup(short_msg)}[/]")
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
        if not is_round_complete:
            if is_current_round:
                t.add_row(get_progress_bar("  Received updates:", n_fog_c, n_fog_c))
            
            t.add_row(f"  {calc_status} [dim white]{calc_text}[/]")
            t.add_row(f"  {save_status} [dim white]{save_text}[/]")
            
    elif has_pending_phase and highest_phase < 7 and is_current_round:
        t.add_row("[dim white]▶ Upcoming: Completing Evaluation...[/]")
    else:
        t.add_row("[bold white]Evaluating Global Model....[/]")
        t.add_row(get_progress_bar("  Received updates:", len(state["fog_clients_delivered"]), n_fog_c))
        
        t.add_row(f"  {calc_status} [dim white]{calc_text}[/]")
        t.add_row(f"  {save_status} [dim white]{save_text}[/]")
        
        has_pending_phase = True

    style = "dim" if p7_complete and not is_current_round else ""
    title_color = "bold white" if is_current_round else "dim grey50"
    
    return Panel(t, title=f"[{title_color}]Round {r_num}/{max_rounds}[/]", title_align="left", border_style="grey37", style=style, box=box.SQUARE)

def generate_dashboard(logs, printed_rounds=None):
    """Build TUI dashboard with all rounds rendered in chronological order."""
    if printed_rounds is None:
        printed_rounds = set()
        
    if not logs:
        return [], Panel("[dim white]No log files discovered.[/]", box=box.SQUARE)

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
        return [], Panel("[dim white]Awaiting federated round initialization...[/]", box=box.SQUARE)

    max_rounds = max(rounds_data.keys())
    
    completed_panels = []
    main_table = Table.grid(expand=True)
    pipeline_completed = False

    # Calculate totals for completion validation
    n_clouds = max(1, len(topology["cloud"]))
    n_fog_c = max(1, len(topology["fog_clients"]))
    n_fog_s = max(1, len(topology["fog_servers"]))
    n_edges = max(1, len(topology["edges"]))
    tot_fog = n_fog_s + n_fog_c

    for r_num in range(1, max_rounds + 1):
        # Skip evaluating rounds we've already permanently printed
        if r_num in printed_rounds:
            continue
            
        round_logs = rounds_data[r_num]
        state = parse_round_state(round_logs, topology)
        
        # 1. Validate Phase Completion
        cloud_c = n_clouds if state["cloud_init"] else min(sum(1 for n in state["nodes_booted"] if "CLOUD" in n), n_clouds)
        fog_c = min(sum(1 for n in state["nodes_booted"] if "FOG" in n), tot_fog)
        edge_c = min(sum(1 for n in state["nodes_booted"] if "EDGE" in n), n_edges)
        
        p1 = (cloud_c == n_clouds) and (fog_c == tot_fog) and (edge_c == n_edges)
        p2 = (len(state["fog_clients_started"]) == n_fog_c) and (len(state["fog_servers_started"]) == n_fog_s)
        p3 = (len(state["edges_trained"]) == n_edges)
        p4 = (len(state["edge_tokens_generated"]) == n_edges)
        
        total_processed_tokens = len(state["edge_tokens_verified"]) + len(state["edge_tokens_rejected"])
        p5 = (total_processed_tokens == n_edges)
        
        p6 = (len(state["fog_servers_aggregated"]) == n_fog_s)
        p7 = state["cloud_evaluated"]
        
        # 2. Strict Boolean Check for completion
        is_fully_complete = all([p1, p2, p3, p4, p5, p6, p7])
        
        # 3. Check for any security/attestation rejections
        has_rejections = len(state["edge_tokens_rejected"]) > 0
        
        # 4. Dynamic Rendering
        is_latest = (r_num == max_rounds)
        should_expand = not is_fully_complete or has_rejections or is_latest
        
        if should_expand:
            main_table.add_row(build_round_panel(r_num, max_rounds, state, topology, is_expanded=True))
        else:
            panel = Panel(f"[dim green]✔ Round {r_num} Completed[/]", border_style="grey37", box=box.SQUARE)
            completed_panels.append((r_num, panel))
            main_table.add_row(panel)
            
        if is_fully_complete and is_latest:
            pipeline_completed = True

    if pipeline_completed:
        main_table.add_row("")
        main_table.add_row(Panel("[bold white]✦ Pipeline Execution Completed Successfully ✦[/]", border_style="white", box=box.SQUARE))

    return completed_panels, main_table


# Presentation-only helpers. Never feed sanitized text back into the log parser.
_ANSI_ESCAPE = re.compile(
    r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC: title / hyperlink
    r"|\x1b[P^_X].*?\x1b\\"                # DCS / PM / APC / SOS
    r"|\x1b\[[0-?]*[ -/]*[@-~]"             # CSI: color / cursor / erase
    r"|\x1b[@-_]",                          # Other ESC commands
    re.DOTALL,
)
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def safe_markup(value):
    """Display log data literally, without allowing it to control the terminal."""
    text = _ANSI_ESCAPE.sub("", str(value))
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return escape(_CONTROL_CHARS.sub("", text))


class ASCIIView:
    """Optional encoding fallback; the normal Unicode design stays unchanged."""

    _translation = str.maketrans({
        "\u2500": "-", "\u2501": "=", "\u2502": "|", "\u2503": "|",
        "\u250c": "+", "\u2510": "+", "\u2514": "+", "\u2518": "+",
        "\u251c": "+", "\u2524": "+", "\u252c": "+", "\u2534": "+", "\u253c": "+",
        "\u2578": ">", "\u257a": "-", "\u2714": "+", "\u2716": "x",
        "\u25b6": ">", "\u25ef": "o", "\u2726": "*", "\u2026": ".",
    })

    def __init__(self, renderable):
        self.renderable = renderable

    def __rich_console__(self, render_console, options):
        for segment in render_console.render(self.renderable, options):
            text = segment.text.translate(self._translation)
            text = text.encode("ascii", "replace").decode("ascii")
            yield Segment(text, segment.style, segment.control)


def stdout_is_tty():
    # Check the real stream, not FORCE_COLOR / TTY_COMPATIBLE overrides.
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, OSError):
        return False


def make_console(plain=False):
    capable = stdout_is_tty() and os.environ.get("TERM", "").lower() not in {
        "dumb", "unknown"
    }
    return Console(
        file=sys.stdout,
        force_terminal=None if capable else False,
        force_jupyter=False,
        color_system=None if plain or not capable else "auto",
        safe_box=True,
        emoji=False,
    )


def update_width(render_console, requested_width=None):
    """Refresh the real terminal width; stale COLUMNS must not cause wrapping."""
    width = None
    if stdout_is_tty():
        try:
            width = os.get_terminal_size(sys.stdout.fileno()).columns
        except (AttributeError, OSError, ValueError):
            pass
    if width is None or width < 1:
        width = requested_width or render_console.width
    elif requested_width is not None:
        width = min(width, requested_width)
    render_console.width = max(1, width)
    # Height remains automatic except when the actual TTY can be queried.
    if stdout_is_tty():
        try:
            height = os.get_terminal_size(sys.stdout.fileno()).lines
            if height > 0:
                render_console.height = height
        except (AttributeError, OSError, ValueError):
            pass


def full_dashboard(logs, ascii_only=False):
    """Compose all round panels in chronological order."""
    _, view = generate_dashboard(logs)
    return ASCIIView(view) if ascii_only else view


def render_snapshot(render_console, renderable):
    """Render one immutable frame for printing and change detection."""
    with render_console.capture() as capture:
        render_console.print(renderable)
    return capture.get()


def write_snapshot(render_console, frame):
    # No cursor movement, screen erasure, or carriage-return repainting here.
    render_console.file.write(frame)
    render_console.file.flush()


def watch_snapshots(render_console, interval=1.0, width=None, ascii_only=False):
    """Append changed frames only. Previously printed lines are never redrawn."""
    previous_frame = None
    while True:
        update_width(render_console, width)
        frame = render_snapshot(render_console, full_dashboard(load_logs(), ascii_only))
        if frame != previous_frame:
            write_snapshot(render_console, frame)
            previous_frame = frame
        time.sleep(interval)


def watch_fullscreen(render_console, interval=1.0, width=None, ascii_only=False):
    """Explicit one-screen view; Ctrl+C restores the prompt and prints a snapshot."""
    update_width(render_console, width)
    latest_logs = load_logs()
    latest_view = full_dashboard(latest_logs, ascii_only)
    live = Live(
        latest_view,
        console=render_console,
        screen=True,
        transient=True,
        auto_refresh=False,
        vertical_overflow="ellipsis",
    )
    try:
        try:
            # Explicit finally also handles interrupts DURING start/first render,
            # before a context manager's __enter__ would have returned.
            live.start(refresh=True)
            while True:
                update_width(render_console, width)
                latest_logs = load_logs()
                latest_view = full_dashboard(latest_logs, ascii_only)
                live.update(latest_view, refresh=True)
                time.sleep(interval)
        finally:
            live.stop()
            # Also restore if a signal arrived while alternate-screen entry was
            # being written, before Live recorded its successful transition.
            render_console.show_cursor(True)
            if render_console.is_alt_screen:
                render_console.set_alt_screen(False)
    except KeyboardInterrupt:
        # Live has already restored cursor visibility and the normal screen.
        update_width(render_console, width)
        final_view = full_dashboard(latest_logs, ascii_only)
        write_snapshot(render_console, render_snapshot(render_console, final_view))


@contextmanager
def graceful_termination():
    """Let Live's context manager restore terminal state on normal SIGTERM."""
    previous = signal.getsignal(signal.SIGTERM)

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def positive_seconds(value):
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("interval must be a positive number") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("interval must be a finite positive number")
    return seconds


def positive_width(value):
    try:
        width = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("width must be a positive integer") from exc
    if width < 1:
        raise argparse.ArgumentTypeError("width must be a positive integer")
    return width


def main():
    global LOG_DIR, console
    parser = argparse.ArgumentParser(
        description="Read pipeline logs using the original dashboard and phase rules.",
        epilog=(
            "Watch mode appends changed snapshots for safe terminal scrollback. "
            "Full-screen mode is an optional bounded viewport, not scrollback."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "-w", "--watch", action="store_true",
        help="watch using append-only changed snapshots (scroll-safe)",
    )
    mode.add_argument(
        "--live", action="store_true",
        help="watch in the alternate screen; use --watch for full scrollback",
    )
    parser.add_argument("--interval", type=positive_seconds, default=1.0,
                        help="watch refresh interval in seconds (default: 1)")
    parser.add_argument("--width", type=positive_width,
                        help="maximum output width; clamped to the real TTY width")
    parser.add_argument("--log-dir", default=LOG_DIR,
                        help="log directory (default: logs/nodes, relative to cwd)")
    parser.add_argument("--plain", action="store_true",
                        help="disable all ANSI color/style output")
    parser.add_argument("--ascii", action="store_true",
                        help="use ASCII fallback glyphs without changing phase logic")
    args = parser.parse_args()
    LOG_DIR = args.log_dir
    console = make_console(args.plain)

    # Keep the existing Unicode graphics unless the stream cannot encode them.
    ascii_only = args.ascii
    if not ascii_only:
        try:
            "\u250c\u2500\u2502\u2714\u2716\u25b6\u25ef\u2726".encode(
                getattr(sys.stdout, "encoding", None) or "utf-8")
        except (UnicodeEncodeError, LookupError):
            ascii_only = True

    interactive = (stdout_is_tty() and console.is_terminal
                   and not console.is_dumb_terminal and not console.is_jupyter
                   and not console.legacy_windows and console.is_interactive)
    try:
        with graceful_termination():
            if args.live and interactive:
                watch_fullscreen(console, args.interval, args.width, ascii_only)
            elif args.watch or args.live:
                if args.live:
                    print("No interactive terminal: using scroll-safe snapshots.", file=sys.stderr)
                watch_snapshots(console, args.interval, args.width, ascii_only)
            else:
                update_width(console, args.width)
                write_snapshot(console, render_snapshot(console, full_dashboard(load_logs(), ascii_only)))
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        # E.g. '| head': avoid a traceback and a second failure during shutdown.
        sys.stdout = open(os.devnull, "w")


if __name__ == "__main__":
    main()