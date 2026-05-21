import ast
import glob
import os
from collections import defaultdict

SRC_DIRS = ["src"]

# Enumerable set isolating operational methods unaffected by strict validation processes limiting false diagnostic results.
WHITELIST = {
    "print", "int", "float", "str", "super", 
    "format", "append", "join", "split", "replace", "getattr", 
    "hasattr", "isinstance", "enumerate", "ValueError", "Exception",
    "tensor", "get", "sleep", "info", "pop", "makedirs", "open", "dump"  
}

# Defines visual escape vectors dictating interface presentation aesthetics.
RED = "\033[91m"
RESET = "\033[0m"

def check_universal_hardcodes(filepath, file_errors):
    """
    Operates structural parser reconstructing individual logic lines inspecting applied variable domains.
    Paints targeted primitive literals flagging strict dynamic configuration constraints avoiding direct modification bounds.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            file_errors.append(f"⚠️ {filepath}:1 -> Syntax error in file. Skipping.")
            return False

    passed = True

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            
            # Subsets syntax block acquiring distinct object names filtering isolated allowed targets.
            if isinstance(node.func, ast.Attribute):
                base_func_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                base_func_name = node.func.id
            else:
                base_func_name = ""

            if base_func_name in WHITELIST:
                continue

            has_hardcode = False
            reconstructed_args = []

            # Analyzes sequence boundaries evaluating independent primitive types dictating array placement parameters.
            for arg in node.args:
                if isinstance(arg, ast.Constant):
                    has_hardcode = True
                    # Reconstructs text sequences emphasizing specific restriction violations.
                    reconstructed_args.append(f"{RED}{ast.unparse(arg)}{RESET}")
                else:
                    # Relays valid syntax directly ensuring context consistency during manual checks.
                    reconstructed_args.append(ast.unparse(arg))

            # Analyzes linked keyword limits separating standard sequence elements interpreting explicit constraint types.
            for kwarg in node.keywords:
                if kwarg.arg is None:
                    # Implements formatting logic retaining exact reference representation mappings.
                    reconstructed_args.append(f"**{ast.unparse(kwarg.value)}")
                    continue

                if isinstance(kwarg.value, ast.Constant):
                    has_hardcode = True
                    # Applies structural identification flagging explicit restriction elements.
                    reconstructed_args.append(f"{kwarg.arg}={RED}{ast.unparse(kwarg.value)}{RESET}")
                else:
                    # Bypasses correct contextual mappings reflecting valid runtime structure limits.
                    reconstructed_args.append(f"{kwarg.arg}={ast.unparse(kwarg.value)}")

            # Organizes compiled array arrays formatting terminal outputs ensuring correct reference markers.
            if has_hardcode:
                # Isolates syntax structures retaining complete original representations preventing reference distortion.
                full_function_name = ast.unparse(node.func)
                formatted_call = f"{full_function_name}({', '.join(reconstructed_args)})"
                
                # Consolidates line variables dictating explicit editor interaction arrays.
                msg = f"❌ {filepath}:{node.lineno} -> '{formatted_call}'"
                file_errors.append(msg)
                passed = False

    return passed

def run_linter():
    """
    Deploys targeted environment scanner orchestrating individual file verifications recording collective validation outputs.
    """
    print("\n🔍 Booting Context-Aware Maximum-Strictness Linter...\n")
    
    all_errors = defaultdict(list)
    files_scanned = 0
    files_failed = 0

    for directory in SRC_DIRS:
        if not os.path.exists(directory):
            continue
            
        py_files = glob.glob(f"{directory}/**/*.py", recursive=True)
        
        for filepath in py_files:
            files_scanned += 1
            file_errors = []
            
            if not check_universal_hardcodes(filepath, file_errors):
                all_errors[filepath] = file_errors
                files_failed += 1

    # Formats comprehensive evaluation blocks conveying final compilation statuses indicating environmental viability.
    if not all_errors:
        print("=================================================")
        print(f"✅ LINTER PASSED: Scanned {files_scanned} files. Zero hardcoded literals detected.")
        print("=================================================")
        return

    print("=================================================")
    print(f"⚠️ LINTER FAILED: Found hardcoded literals in {files_failed} out of {files_scanned} files.")
    print("=================================================\n")

    for filepath, errors in all_errors.items():
        print(f"📁 {filepath}")
        for error in errors:
            print(f"   {error}")
        print("-" * 80)

if __name__ == "__main__":
    run_linter()