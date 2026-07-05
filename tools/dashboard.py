import streamlit as st
import pandas as pd
from log_parser import parse_logs
import time
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="ZTA-FL Live Architecture", layout="wide", initial_sidebar_state="expanded")

# --- UTILITIES & UI RENDERERS ---
def render_step_tracker(steps, current_index):
    html = '<div style="display: flex; flex-direction: column; gap: 10px;">'
    for i, step_obj in enumerate(steps):
        title = step_obj["title"]
        details = step_obj.get("details", [])
        live_logs = step_obj.get("live_logs", [])
        
        if i < current_index:
            bg, color, icon = "#00cc96", "black", "✅"
            combined = live_logs + details
        elif i == current_index:
            bg, color, icon = "#1f77b4", "white", "⏳"
            combined = live_logs + details
        else:
            bg, color, icon, combined = "#333333", "#aaaaaa", "⚪", details
            
        if combined:
            tree_html = '<div style="font-family: monospace; color: #bbbbbb; margin-left: 18px; margin-top: 5px; font-size: 0.85rem; line-height: 1.4;">|<br>'
            for j, detail in enumerate(combined):
                tree_html += f'&lfloor;&nbsp;&nbsp;{detail}<br>' if j == len(combined)-1 else f'|-- {detail}<br>'
            tree_html += '</div>'
        else:
            tree_html = ""
            
        html += f"""<div>
<div style="background-color: {bg}; color: {color}; padding: 12px; border-radius: 5px; font-weight: bold; border: 1px solid #555;">
{icon} &nbsp; {title}
</div>
{tree_html}
</div>"""
    html += '</div>'
    return html

def get_color_dict(status):
    if status == "Round Complete":
        return {"fill": "#00cc96", "stroke": "#00cc96", "text": "#000000"} 
    elif status in ["Idle", "Idle / Awaiting Cloud", "UNKNOWN"]:
        return {"fill": "#333333", "stroke": "#aaaaaa", "text": "#aaaaaa"} 
    elif status == "Rejected":
        return {"fill": "#ff4b4b", "stroke": "#ff4b4b", "text": "#ffffff"} 
    else:
        return {"fill": "#1f77b4", "stroke": "#1f77b4", "text": "#ffffff"} 

def render_full_network_svg(network_state):
    nodes_state = network_state["nodes"]
    cloud_status = network_state.get("cloud_status", "Idle")
    c_cloud = get_color_dict(cloud_status)
    
    svg = f"""<svg viewBox="0 0 1000 400" width="100%" height="400" xmlns="http://www.w3.org/2000/svg" style="background-color: transparent; font-family: sans-serif; font-weight: bold;">\n"""

    fog_nodes = {}
    for n in nodes_state.values():
        if n["Type"] == "fog":
            m = re.search(r"FOG (\d+)", n["ID"])
            if m:
                fog_nodes[m.group(1)] = n["ID"]
                
    if not fog_nodes:
        return svg + "</svg>"

    edge_nodes = [n for n in nodes_state.values() if n["Type"] == "edge"]
    fog_to_edges = {f_num: [] for f_num in fog_nodes.keys()}
    
    for e in edge_nodes:
        m = re.search(r"\[EDGE (\d+)_", e["ID"])
        if m and m.group(1) in fog_to_edges:
            fog_to_edges[m.group(1)].append(e)

    svg += f"""<rect x="420" y="30" width="160" height="40" rx="8" fill="{c_cloud['fill']}" stroke="{c_cloud['stroke']}" stroke-width="2"/>
<text x="500" y="55" fill="{c_cloud['text']}" text-anchor="middle" dominant-baseline="middle" font-size="16">☁️ Cloud Server</text>\n"""

    f_keys = sorted(list(fog_nodes.keys()))
    spacing = 1000 / (len(f_keys) + 1)
    
    for i, f_num in enumerate(f_keys):
        fx = spacing * (i + 1)
        fog = nodes_state[fog_nodes[f_num]]
        c_fog = get_color_dict(fog["Status"])
        
        if fog["Status"] == "Round Complete":
            cf_line = "#00cc96" 
        elif fog["Status"] in ["Idle", "Idle / Awaiting Cloud", "UNKNOWN"]:
            cf_line = "#1f77b4" if cloud_status in ["Broadcasting", "Aggregating"] else "#555555" 
        else:
            cf_line = "#1f77b4" 
            
        svg += f'<line x1="500" y1="70" x2="{fx}" y2="180" stroke="{cf_line}" stroke-width="4" />\n'
        
        svg += f"""<rect x="{fx - 70}" y="180" width="140" height="36" rx="8" fill="{c_fog['fill']}" stroke="{c_fog['stroke']}" stroke-width="2"/>
<text x="{fx}" y="200" fill="{c_fog['text']}" text-anchor="middle" dominant-baseline="middle" font-size="14">🌫️ Fog Node {f_num}</text>\n"""
        
        edges = fog_to_edges[f_num]
        m = len(edges)
        for j, edge in enumerate(edges):
            ex = fx - (m - 1) * 45 + j * 90 
            c_edge = get_color_dict(edge["Status"])
            
            if edge["Status"] == "Round Complete":
                fe_line = "#00cc96" 
            elif edge["Status"] == "Rejected":
                fe_line = "#ff4b4b" 
            elif edge["Status"] in ["Idle", "UNKNOWN"]:
                fe_line = "#1f77b4" if fog["Status"] in ["Propagating to Edges", "Receiving Tokens"] else "#555555" 
            else:
                fe_line = "#1f77b4" 
                
            svg += f'<line x1="{fx}" y1="216" x2="{ex}" y2="320" stroke="{fe_line}" stroke-width="3" />\n'
            
            is_mal = "Byzantine" in edge.get("Role", "")
            strk = "#ff4b4b" if is_mal else c_edge["stroke"]
            strkw = "3" if is_mal else "2"
            icon = "☠️" if is_mal else "📱"
            
            edge_id_match = re.search(r"EDGE (\d+_\d+)", edge["ID"])
            edge_lbl = f"E-{edge_id_match.group(1)}" if edge_id_match else "Edge"
            
            svg += f"""<rect x="{ex-35}" y="320" width="70" height="30" rx="5" fill="{c_edge['fill']}" stroke="{strk}" stroke-width="{strkw}"/>
<text x="{ex}" y="335" fill="{c_edge['text']}" text-anchor="middle" dominant-baseline="middle" font-size="12">{icon} {edge_lbl}</text>\n"""

    return svg + "</svg>"

cloud_steps = [
    {"title": "Initialize Global Weights (θ_t)", "details": ["Architecture: 8-bit quantized CNN-LSTM"]},
    {"title": "Broadcast Weights to Fog Nodes", "details": ["Protocol: Distributes updated policies downwards"]},
    {"title": "Await Filtered Updates from Fogs", "details": ["Latency allowance: Δt_max = 60s"]},
    {"title": "Perform Global Aggregation (θ_t+1)", "details": ["Formula: θ_t+1 = Σ (w_f * θ_f_t)"]}
]

# --- MAIN DASHBOARD LOGIC ---
st.title("ZTA-FL Prototype: Live Telemetry")

state_history = parse_logs()
if not state_history:
    st.warning("No logs detected in the target directory.")
    st.stop()

available_rounds = sorted(list(state_history.keys()))

with st.sidebar:
    st.header("⚙️ Telemetry Controls")
    view_mode = st.radio("Time View", ["Live (Latest)", "Historical Round"])
    
    if view_mode == "Historical Round":
        selected_round = st.slider("Select Round Snapshot", min_value=min(available_rounds), max_value=max(available_rounds), value=max(available_rounds))
        auto_refresh = False
    else:
        selected_round = max(available_rounds)
        auto_refresh = st.checkbox("Enable Live Auto-Refresh", value=True)
        if st.button("🔄 Manual Refresh"):
            st.rerun()

network_state = state_history[selected_round]
parsed_params = network_state.get("params", {})
nodes_df = pd.DataFrame(network_state["nodes"].values()) if network_state["nodes"] else pd.DataFrame()

tab_net, tab_cloud, tab_fog, tab_edge = st.tabs(["🌐 Whole Network", "☁️ Cloud Node", "🌫️ Fog Node (Aggregator)", "📱 Edge Node (Client)"])

with tab_net:
    st.subheader(f"Network Topology Map (Snapshot: Round {selected_round})")
    st.markdown(
        '<div style="display: flex; gap: 20px; padding: 12px; background-color: #1e1e1e; border-radius: 5px; margin-bottom: 15px; border: 1px solid #444; font-size: 14px; align-items: center; flex-wrap: wrap;">'
        '<span style="color: #ffffff; font-weight: bold; margin-right: 10px;">Legend:</span>'
        '<div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 50%; background-color: #333333; border: 1px solid #aaaaaa;"></div><span>Gray: Pending/Idle</span></div>'
        '<div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 50%; background-color: #1f77b4;"></div><span>Blue: Active/Waiting on Task</span></div>'
        '<div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 50%; background-color: #00cc96;"></div><span>Green: Round Completed</span></div>'
        '<div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 50%; background-color: #ff4b4b;"></div><span>Red Fill: Rejected Payload</span></div>'
        '<div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 50%; background-color: #333333; border: 2px solid #ff4b4b;"></div><span>Red Border: Byzantine Threat</span></div>'
        '</div>', 
        unsafe_allow_html=True
    )
    
    st.markdown(render_full_network_svg(network_state), unsafe_allow_html=True)
    
    with st.expander("📡 Network & Dataset Parameters", expanded=False):
        if parsed_params.get("network"):
            st.json(parsed_params["network"])
        else:
            st.info("Network parameters not yet extracted from logs.")

    if not nodes_df.empty:
        st.dataframe(nodes_df[["ID", "Role", "Status"]], use_container_width=True, hide_index=True)

with tab_cloud:
    cloud_kpi = st.columns(3)
    cloud_kpi[0].metric("Current Global Round", selected_round)
    cloud_kpi[1].metric("Cloud Status", network_state["cloud_status"])
    
    agg_strategy = parsed_params.get("cloud", {}).get("strategy", "Unknown").strip('"\'').upper()
    cloud_kpi[2].metric("Aggregation Strategy", agg_strategy)
    
    st.markdown("### Cloud Process Pipeline")
    
    cloud_idx = 0
    if network_state["cloud_status"] == "Broadcasting": cloud_idx = 1
    elif network_state["cloud_status"] == "Aggregating": cloud_idx = 3
    elif network_state["cloud_status"] == "Round Complete": cloud_idx = 4
        
    dynamic_cloud_steps = []
    for step in cloud_steps:
        dynamic_cloud_steps.append({"title": step["title"], "details": list(step["details"])})
        
    if agg_strategy == "ZTA":
        accuracy = network_state.get("cloud_accuracy")
        rollback_thresh_str = parsed_params.get("cloud", {}).get("rollback_threshold", "0.0").strip('"\'')
        
        if accuracy is not None:
            try:
                rollback_thresh = float(rollback_thresh_str)
                if accuracy < rollback_thresh:
                    acc_text = f"Accuracy: {accuracy} < {rollback_thresh} - Rolling back due to low accuracy"
                else:
                    acc_text = f"Accuracy: {accuracy} >= {rollback_thresh} - Accuracy accepted"
                dynamic_cloud_steps[-1]["details"].append(acc_text)
            except ValueError:
                dynamic_cloud_steps[-1]["details"].append(f"Accuracy: {accuracy}")
        elif network_state["cloud_status"] == "Round Complete":
            dynamic_cloud_steps[-1]["details"].append("Accuracy: Awaiting logs or failed to parse")
    else:
        dynamic_cloud_steps[-1]["details"] = []
            
    st.markdown(render_step_tracker(dynamic_cloud_steps, cloud_idx), unsafe_allow_html=True)

with tab_fog:
    st.markdown("### Fog Nodes Overview")
    if not nodes_df.empty:
        fog_df = nodes_df[nodes_df["Type"] == "fog"]
        if not fog_df.empty:
            st.dataframe(fog_df[["ID", "Status"]], use_container_width=True, hide_index=True)
            
    st.markdown("---")

    fog_nodes = [n["ID"] for n in network_state["nodes"].values() if n["Type"] == "fog"]
    if fog_nodes:
        fog_selector = st.selectbox("Inspect Fog Node:", fog_nodes)
        target_fog = network_state["nodes"][fog_selector]
        
        col1, col2 = st.columns(2)
        col1.metric("Node ID", target_fog["ID"])
        col2.metric("Current Activity", target_fog["Status"])
        
        f_col_left, f_col_right = st.columns([1, 1.5])
        
        with f_col_left:
            st.markdown("### Fog Process Pipeline")
            f_num_match = re.search(r"FOG (\d+)", target_fog["ID"])
            f_num = f_num_match.group(1) if f_num_match else "0"
            
            fog_edges = [n for n in network_state["nodes"].values() if n["Type"] == "edge" and f"EDGE {f_num}_" in n["ID"]]
            total_edges = len(fog_edges)
            
            fog_edge_ids = [e["ID"] for e in fog_edges]
            current_round_tokens = len([t for t in network_state["tokens"] if t["Round"] == selected_round and t["Edge_ID"] in fog_edge_ids])
            
            quarantined_count = len([e for e in fog_edges if e["Status"] == "Rejected"])
            successful_verifications = max(0, current_round_tokens - quarantined_count)
            
            accepted_count = len([e for e in fog_edges if e["Status"] == "Round Complete"])
            dropped_shap = max(0, successful_verifications - accepted_count)

            shap_thresh = parsed_params.get("fog", {}).get("shap_threshold", "0.5")
            shap_samples = parsed_params.get("fog", {}).get("shap_val_samples", "100")
            shap_count = parsed_params.get("fog", {}).get("shap_explain_count", "10")

            dynamic_fog_steps = [
                {"title": "Await Cloud Broadcast", "details": ["Receiving θ_t from Cloud Layer"]},
                {"title": "Propagate Weights to Assigned Edges", "details": [f"Reached {total_edges} edge nodes"]},
                {"title": "Receive TPM Tokens & Edge Updates", "details": [f"Received {current_round_tokens} tokens"]},
                {"title": "Execute Cryptographic Verification & TrustDB Check", "details": [f"{successful_verifications} Successful verification(s), {quarantined_count} Node(s) quarantined"]},
                {"title": "Calculate SHAP Stability & Aggregate", "details": [
                    f"The SHAP threshold is set to {shap_thresh}.",
                    f"The model utilizes {shap_samples} SHAP validation samples.",
                    f"The SHAP explain count is configured to {shap_count}.",
                    f"Accepted {accepted_count} edge nodes out of {total_edges}, Dropped {dropped_shap} verified nodes."
                ]},
                {"title": "Sending updates to Cloud", "details": ["Relaying aggregated weights to Global Cloud"]}
            ]
            
            fog_idx = 0
            if target_fog["Status"] == "Propagating to Edges": fog_idx = 1
            elif target_fog["Status"] == "Receiving Tokens": fog_idx = 2
            elif target_fog["Status"] == "Verifying & TrustDB Check": fog_idx = 3
            elif target_fog["Status"] == "SHAP Aggregation": fog_idx = 4
            elif target_fog["Status"] == "Round Complete": fog_idx = 6 
            
            st.markdown(render_step_tracker(dynamic_fog_steps, fog_idx), unsafe_allow_html=True)

        with f_col_right:
            st.markdown(f"### Logs for {fog_selector} (Round {selected_round})")
            for log in target_fog["Logs"]:
                st.code(log, language="log")
    else:
        st.info("No Fog Nodes detected in the logs yet.")

    st.markdown("---")
    st.markdown("### Global TrustDB State")
    if network_state["trust_db"]:
        st.dataframe(pd.DataFrame(network_state["trust_db"]).T, use_container_width=True)
    else:
        st.info("No TrustDB rewards/penalties registered in current logs.")

with tab_edge:
    st.markdown("### Edge Nodes Overview")
    if not nodes_df.empty:
        edge_df = nodes_df[nodes_df["Type"] == "edge"]
        if not edge_df.empty:
            st.dataframe(edge_df[["ID", "Role", "Status"]], use_container_width=True, hide_index=True)
            
    st.markdown("---")

    edge_nodes = [n["ID"] for n in network_state["nodes"].values() if n["Type"] == "edge"]
    if edge_nodes:
        edge_selector = st.selectbox("Inspect Node:", edge_nodes)
        target_node = network_state["nodes"][edge_selector]
        
        col1, col2 = st.columns(2)
        col1.metric("Node Role", target_node["Role"])
        col2.metric("Current Activity", target_node["Status"])
        
        c_edge_left, c_edge_right = st.columns([1, 1.5])
        
        with c_edge_left:
            st.markdown("### Edge Process Pipeline")
            
            # Retrieve the newly parsed live parameters from the node's state
            rp = target_node.get("runtime_params", {})
            r_type = rp.get("type", "UNKNOWN")
            
            # Base Local Training Configuration
            training_details = [
                "Architecture: 8-bit CNN-LSTM (h_t = LSTM(CNN(x_t), h_{t-1}))"
            ]
            
            if "lr" in rp:
                training_details.append(f"Learning Rate: {rp.get('lr')}, Epochs: {rp.get('epochs')}")
            if "num_classes" in rp:
                training_details.append(f"Number of classes: {rp.get('num_classes')}, Clip norm: {rp.get('clip_norm')}")

            threat_details = []
            
            # Display matching variables according to the specific logged operation
            if r_type == "SHAP AWARE":
                threat_details.append(f"Attack Type: SHAP AWARE Attack ({rp.get('shap_aware_base_attack', 'N/A')})")
                threat_details.append(f"SHAP Threshold: {rp.get('shap_threshold', 'N/A')}, Explain Count: {rp.get('shap_explain_count', 'N/A')}, Val Samples: {rp.get('shap_val_samples', 'N/A')}")
                threat_details.append(f"Alpha scale for gradient manip: {rp.get('alpha_scale', 'N/A')}, Flip probability (p_flip): {rp.get('p_flip', 'N/A')}")
                
            elif r_type == "BACKDOOR":
                threat_details.append("Attack Type: Backdoor Attack")
                threat_details.append(f"Backdoor poison fraction: {rp.get('backdoor_poison_fraction', 'N/A')}, Target class: {rp.get('target_class', 'N/A')}")
                threat_details.append(f"Backdoor Trigger value: {rp.get('trigger_value', 'N/A')}, Trigger features: {rp.get('trigger_features', 'N/A')}")
                
            elif r_type == "PGD":
                threat_details.append("Attack Type: PGD Attack")
                threat_details.append(f"PGD adv ratio: {rp.get('adv_ratio', 'N/A')}, PGD number of iter: {rp.get('pgd_n_iter', 'N/A')}")
                threat_details.append(f"PGD eps: {rp.get('eps', 'N/A')}, PGD alpha: {rp.get('alpha', 'N/A')}")
                threat_details.append(f"PGD clip_min: {rp.get('clip_min', 'N/A')}, PGD clip_max: {rp.get('clip_max', 'N/A')}")
                
            elif r_type == "FGSM":
                threat_details.append("Attack Type: FGSM Attack")
                threat_details.append(f"FGSM adv ratio: {rp.get('adv_ratio', 'N/A')}, FGSM eps: {rp.get('eps', 'N/A')}")
                threat_details.append(f"FGSM clip_min: {rp.get('clip_min', 'N/A')}, FGSM clip_max: {rp.get('clip_max', 'N/A')}")
                
            elif r_type == "BENIGN":
                threat_details.append("Defense: Adversarial Training on Benign Node")
                threat_details.append(f"Adversary Ratio: {rp.get('adv_ratio', 'N/A')}, Epsilon: {rp.get('eps', 'N/A')}, Alpha: {rp.get('alpha', 'N/A')}")
                threat_details.append(f"Number of Iter: {rp.get('n_iter', 'N/A')}, Use pgd (if false, use fgsm): {rp.get('use_pgd', 'N/A')}")
                threat_details.append(f"clip_min: {rp.get('clip_min', 'N/A')}, clip_max: {rp.get('clip_max', 'N/A')}")
                
            elif r_type != "UNKNOWN":
                threat_details.append(f"Attack Type: {r_type} Attack")
                threat_details.append(f"Alpha scale: {rp.get('alpha_scale', 'N/A')}, p_flip: {rp.get('p_flip', 'N/A')}")
                
            else:
                threat_details.append("Awaiting execution logs to parse runtime parameters...")
                
            # Final Attestation Details
            attestation_details = [
                "Tokens generated: {ID_{i,t}, PCR, Sig_{TPM}}",
                "Destination: Broadcasting token bundle to Fog Layer"
            ]

            # Render Pipeline
            dynamic_edge_steps = [
                {"title": "Local Intrusion Detection System (Local IDS)", "details": training_details},
                {"title": "Adversarial Training & Role Execution", "details": threat_details},
                {"title": "TPM-Based Attestation Module", "details": attestation_details}
            ]
            
            # Identify correct UI phase based on current logs
            status = target_node["Status"]
            edge_idx = 0
            
            if "Data Preparation" in status or "Training" in status:
                edge_idx = 0
                if "Complete" in status:
                    edge_idx = 1
            elif "Generating Token" in status:
                edge_idx = 2
            elif status in ["Round Complete", "Rejected"]:
                edge_idx = 3
                
            st.markdown(render_step_tracker(dynamic_edge_steps, edge_idx), unsafe_allow_html=True)
            
        with c_edge_right:
            st.markdown(f"### Logs for {edge_selector} (Round {selected_round})")
            for log in target_node["Logs"]:
                st.code(log, language="log")
                
            st.markdown("### Generated TPM Tokens")
            node_tokens = [t for t in network_state["tokens"] if t["ID_i"] == edge_selector and t["Round"] <= selected_round]
            if node_tokens:
                # Remove internal tracking columns prior to rendering
                display_df = pd.DataFrame(node_tokens).drop(columns=["Edge_ID", "Round"], errors="ignore")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No TPM tokens observed for this node yet.")
            
    else:
        st.info("No Edge Nodes detected in the logs yet.")

# Auto-refresh loop handler
if auto_refresh:
    time.sleep(1)
    st.rerun()