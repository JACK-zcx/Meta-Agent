import xml.etree.ElementTree as ET


def _parse_single_agent(agent_elem):
    """Parse a single <Agent> element into a dict."""
    result = {
        'Agent Name': None,
        'Agent Description': None,
        'Agent Query': None,
        'Tasks': [],
        'Tool': None,
        'Answer': None
    }
    agent_name = agent_elem.find("./Agent-Name")
    if agent_name is not None:
        result['Agent Name'] = agent_name.text.strip() if agent_name.text else None

    agent_description = agent_elem.find("./Agent-Description")
    if agent_description is not None:
        result['Agent Description'] = agent_description.text.strip() if agent_description.text else None

    agent_query = agent_elem.find("./Agent-Query")
    if agent_query is not None:
        result['Agent Query'] = agent_query.text.strip() if agent_query.text else None

    tasks = agent_elem.findall("./Tasks/Task")
    for task in tasks:
        if task is not None and task.text:
            result['Tasks'].append(task.text.strip())

    tool = agent_elem.find("./Tool")
    if tool is not None:
        tool_info = {}
        tool_name = tool.find("Tool-Name")
        tool_description = tool.find("Tool-Description")
        if tool_name is not None and tool_name.text:
            tool_info['Tool Name'] = tool_name.text.strip()
        if tool_description is not None and tool_description.text:
            tool_info['Tool Description'] = tool_description.text.strip()
        result['Tool'] = tool_info

    return result


def extract_from_xml(xml_string):
    """
    Parse XML from MetaAgent's LLM response.

    Supports:
    - Single <Agent> block → returns single dict
    - Multiple <Agent> blocks (parallel mode) → returns dict with 'agents' list and 'parallel': True
    - <Final-Answer> block → returns dict with Answer key
    """
    # Strip leading/trailing whitespace
    xml_string = xml_string.strip()

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        # Try wrapping in a root element for multi-agent case
        try:
            root = ET.fromstring(f"<Root>{xml_string}</Root>")
        except ET.ParseError as e:
            return {"error": str(e)}

    # Case 1: <Final-Answer> root tag
    if root.tag == "Final-Answer":
        return {'Agent Name': None, 'Agent Description': None,
                'Agent Query': None, 'Tasks': [], 'Tool': None,
                'Answer': (root.text or '').strip()}

    # Case 2: Single or Multiple <Agent>
    # Check if root itself is an Agent element
    if root.tag == "Agent":
        agent_elems = [root] + root.findall(".//Agent")
    else:
        agent_elems = root.findall(".//Agent")
    if not agent_elems:
        return {"error": "No <Agent> or <Final-Answer> found in XML"}

    agents = [_parse_single_agent(a) for a in agent_elems]

    if len(agents) == 1:
        result = agents[0]
        result['parallel'] = False
        return result

    # Case 3: Multiple agents → parallel execution
    return {
        'agents': agents,
        'parallel': True,
        'Agent Name': ', '.join(a['Agent Name'] for a in agents if a['Agent Name']),
        'Tasks': [t for a in agents for t in a['Tasks']],
    }


def read_markdown_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    return markdown_content
