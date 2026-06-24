import ast
from textwrap import dedent

def extract_tools_from_module(location:str):
    with open(location, 'r', encoding='utf-8') as f:
        tree=ast.parse(f.read())
    tool_definitions=[]
    models_defintions=[]
    tool_names=[]
    func_names=[]
    tools=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef):
            func_names.append(node.name)
            tool_definitions.append(ast.unparse(node))
            for decorator in node.decorator_list:
                if isinstance(decorator,ast.Call):
                    tool_names.append(ast.unparse(decorator.args))
        if isinstance(node,ast.ClassDef):
            models_defintions.append(ast.unparse(node))
    for i in range(len(tool_names)):
        tools.append({'tool_name':tool_names[i].replace('\'',''),
        'tool':models_defintions[i]+'\n\n'+tool_definitions[i],
        'func_name':func_names[i]})
    return tools

def update_tool_to_module(location: str, tool_code: str):
    """Replace existing tool code in the module file. tool_code is the new source code."""
    with open(location, 'r', encoding='utf-8') as f:
        module = f.read()
    tree = ast.parse(module)
    nodes = tree.body

    # Parse the new code to find function/class names
    try:
        new_tree = ast.parse(tool_code)
        new_func_name = None
        new_class_name = None
        for node in ast.walk(new_tree):
            if isinstance(node, ast.FunctionDef):
                new_func_name = node.name
            elif isinstance(node, ast.ClassDef):
                new_class_name = node.name
    except Exception:
        # Fallback: just append
        with open(location, 'a', encoding='utf-8') as f:
            f.write(f'{tool_code}\n\n')
        return

    # Replace matching nodes
    for index, node in enumerate(nodes):
        if new_class_name and isinstance(node, ast.ClassDef) and node.name == new_class_name:
            for n in ast.walk(new_tree):
                if isinstance(n, ast.ClassDef):
                    nodes[index] = n
        elif new_func_name and isinstance(node, ast.FunctionDef) and node.name == new_func_name:
            for n in ast.walk(new_tree):
                if isinstance(n, ast.FunctionDef):
                    nodes[index] = n

    with open(location, 'w', encoding='utf-8') as f:
        updated_module = ast.unparse(nodes)
        f.write(f'{updated_module}\n\n')

def remove_tool_from_module(location: str, tool_data: dict):
    # Load the module contents
    with open(location, 'r', encoding='utf-8') as f:
        module = f.read()
    # Parse the module contents into an AST
    tree = ast.parse(module)
    nodes=tree.body
    drop_nodes=[]
    for index,node in enumerate(nodes):
        if isinstance(node, ast.ClassDef) and node.name == tool_data['name'].replace(' Tool',''):
            drop_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == tool_data['tool_name']:
            drop_nodes.append(node)
    nodes=[node for node in nodes if node not in drop_nodes]
        
    # Write the updated module contents back to the file
    with open(location, 'w', encoding='utf-8') as f:
        updated_module = ast.unparse(nodes)
        f.write(f'{updated_module}\n\n')

def save_tool_to_module(location: str, tool_code: str):
    """Append tool source code to the module file."""
    with open(location, 'a', encoding='utf-8') as f:
        f.write(f"{dedent(tool_code)}\n\n")

def read_markdown_file(file_path: str) -> str:
    with open(file_path, 'r',encoding='utf-8') as f:
        markdown_content = f.read()
    return markdown_content