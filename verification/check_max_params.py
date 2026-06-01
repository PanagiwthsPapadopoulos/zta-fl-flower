import ast
import glob
import os

SRC_DIRS = ["src/federation", "src/models", "src/security", "src/utils", "scripts"]

def build_internal_registry(py_files):
    """
    Scans project source structures identifying procedural boundaries outlining argument assignments.
    Builds a complete referential map corresponding to expected parameter counts within customized routines.
    """
    internal_registry = {}
    
    for filepath in py_files:
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read(), filename=filepath)
            except SyntaxError:
                continue
                
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("__") and node.name.endswith("__"):
                    continue
                
                # Iterates syntax arguments building comprehensive lists reflecting default variable assignments.
                all_arg_names = [arg.arg for arg in node.args.args]
                num_defaults = len(node.args.defaults)
                
                if num_defaults > 0:
                    internal_registry[node.name] = {
                        "all_args": all_arg_names,
                        "num_defaults": num_defaults
                    }

    return internal_registry

def check_internal_overrides(filepath, internal_registry):
    """
    Cross-references existing file syntax tracking explicit assignment overlaps blocking incomplete calls.
    Returns status flags identifying leaked default applications.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            return True

    passed = True

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id

            if func_name in internal_registry:
                reg = internal_registry[func_name]
                all_args = reg["all_args"]
                total_args = len(all_args)
                num_defaults = reg["num_defaults"]
                
                # Extracts the specific identifiers representing optional default arguments.
                default_args = all_args[total_args - num_defaults:]
                
                passed_positional_count = len(node.args)
                passed_keywords = [k.arg for k in node.keywords if k.arg is not None]
                
                missing_overrides = []
                
                # Verifies invocation parameters map exclusively verifying comprehensive manual assignment.
                for i, arg_name in enumerate(default_args):
                    # Calculates internal alignment index determining positional offset validity.
                    original_index = (total_args - num_defaults) + i
                    
                    # Approves fulfillment based on positional quantity.
                    covered_by_position = passed_positional_count > original_index
                    
                    # Approves fulfillment based on exact keyword inclusion.
                    covered_by_keyword = arg_name in passed_keywords
                    
                    if not covered_by_position and not covered_by_keyword:
                        missing_overrides.append(arg_name)
                
                if missing_overrides:
                    print(f"❌ [INTERNAL DEFAULT LEAK] {filepath}:{node.lineno})")
                    print(f"   You called '{func_name}()' but left parameters hanging.")
                    print(f"   Missing explicit overrides for: {', '.join(missing_overrides)}")
                    print(f"   Fix: Pass these explicitly.\n")
                    passed = False

    return passed

def run_smart_linter():
    """
    Executes automated structural verification determining compliance mapping rules strictly prohibiting dynamic fallback assignments.
    """
    print("\n🧠 Booting Auto-Discovering Internal Linter...\n")
    
    # Compiles reference list containing operational execution definitions.
    py_files = []
    for directory in SRC_DIRS:
        if os.path.exists(directory):
            py_files.extend(glob.glob(f"{directory}/**/*.py", recursive=True))
            
    # Assembles initial rule structure bounds prior to examination execution.
    internal_registry = build_internal_registry(py_files)
    print(f"🔍 Discovered {len(internal_registry)} internal functions with default variables (ignoring magic methods).")
    print("-" * 50)
    
    # Implements testing methodology passing generated logic nodes sequentially.
    all_passed = True
    for filepath in py_files:
        if not check_internal_overrides(filepath, internal_registry):
            all_passed = False

    print("=================================================")
    if all_passed:
        print(f"✅ LINTER PASSED: No internal defaults are leaking into your pipeline.")
    else:
        print(f"⚠️ LINTER FAILED: Please override the internal variables above.")
    print("=================================================")

if __name__ == "__main__":
    run_smart_linter()