import os
import re
import ast

def get_yaml_val(filepath, key, default):
    """Safely extracts YAML values using regex to avoid external host OS dependencies."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        m = re.search(fr'^{key}:\s*(.+)$', content, re.MULTILINE)
        if m:
            return m.group(1).split('#')[0].strip().strip('"').strip("'")
    except:
        pass
    return default

def main():
    # Inherit PROJECT_ROOT from the parent Bash script
    project_root = os.environ.get("PROJECT_ROOT", ".")
    net_conf = os.path.join(project_root, 'config', 'network.yaml')
    
    try:
        print(f"export CLOUD_SA={get_yaml_val(net_conf, 'cloud_sa_port', '9001')}")
        print(f"export CLOUD_FL={get_yaml_val(net_conf, 'cloud_fl_port', '9002')}")
        print(f"export CLOUD_CTRL={get_yaml_val(net_conf, 'cloud_ctrl_port', '9003')}")
        print(f"export FOG_SA_BASE={get_yaml_val(net_conf, 'fog_sa_base', '9100')}")
        print(f"export FOG_FL_BASE={get_yaml_val(net_conf, 'fog_fl_base', '9200')}")
        print(f"export FOG_CTRL_BASE={get_yaml_val(net_conf, 'fog_ctrl_base', '9300')}")
        print(f"export FOG_CIO_BASE={get_yaml_val(net_conf, 'fog_client_io_base', '9400')}")
        print(f"export EDGE_CIO_BASE={get_yaml_val(net_conf, 'edge_client_io_base', '10000')}")

        num_fogs = int(get_yaml_val(net_conf, 'num_fogs', '2'))
        uniform = int(get_yaml_val(net_conf, 'uniform_edges_per_fog', '2'))
        
        custom_top_str = get_yaml_val(net_conf, 'custom_fog_topology', '[]')
        custom_top = ast.literal_eval(custom_top_str) if custom_top_str else []
        
        edges_array = custom_top[:num_fogs] if custom_top and len(custom_top) >= num_fogs else [uniform] * num_fogs
        
        print(f"export NUM_FOGS={num_fogs}")
        print(f"export EDGES_PER_FOG_ARRAY=({' '.join(map(str, edges_array))})")
    except Exception as e:
        # If it fails, print a bash command that halts the boot sequence
        print(f'echo "Error parsing topology: {e}"; exit 1')

if __name__ == "__main__":
    main()