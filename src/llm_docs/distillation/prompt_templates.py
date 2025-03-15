"""
Prompt templates for documentation distillation.
"""

from typing import Dict, Any
from string import Template

# Base templates
MAIN_PROMPT_TEMPLATE = Template("""
I attached the COMPLETE documentation for ${package_name}, the python library ${package_description}.

The documentation contains a lot of "fluff" and references to things which we do NOT care about. I pasted the docs manually from the documentation website.  

Your goal is to meticulously read and understand everything and then distill it down into the most compact and clear form that would still be easily intelligible to YOU in a new conversation where you had never seen the original documentation text. It's too long to do in one shot, so let's do it in ${num_parts} sections, starting now with the first part. 

Feel free to rearrange the order of sections so that it makes most sense to YOU since YOU will be the one consuming the output! Don't worry if it would be confusing or hard to follow by a human, it's not for a human, it's for YOU to quickly and efficiently convey all the relevant information in the documentation.

${chunk_content}
""")

FOLLOW_UP_PROMPT_TEMPLATE = Template("""
continue with part ${part_num} of ${num_parts}

${chunk_content}
""")

FINAL_PROMPT_TEMPLATE = Template("""
ok now make a part ${part_num} with all the important stuff that you left out from the full docs in the ${prev_parts_count} parts you already wrote above
""")

# Specialized templates for different types of documentation
API_REFERENCE_TEMPLATE = Template("""
I attached the API reference documentation for ${package_name}, the python library ${package_description}.

Focus specifically on summarizing:
1. Function signatures, parameters, return values, and exceptions
2. Class hierarchies and their methods
3. Important constants and configuration options
4. Key usage patterns

Strip away explanatory prose in favor of concise, structured descriptions optimized for an AI assistant to reference quickly. It's too long to do in one shot, so let's do it in ${num_parts} sections, starting now with the first part.

${chunk_content}
""")

TUTORIAL_TEMPLATE = Template("""
I attached tutorial documentation for ${package_name}, the python library ${package_description}.

Focus specifically on extracting:
1. Key code patterns and idioms
2. Common workflows
3. Best practices
4. Important gotchas and edge cases

Distill this into a format optimized for an AI to understand the practical usage patterns, ignoring introductory explanations, context-setting, and verbose examples. It's too long to do in one shot, so let's do it in ${num_parts} sections, starting now with the first part.

${chunk_content}
""")

def get_prompt_for_chunk(package_name: str, chunk_content: str, 
                         part_num: int, num_parts: int, 
                         package_description: str = "",
                         doc_type: str = "general") -> str:
    """
    Get the appropriate prompt for a documentation chunk.
    
    Args:
        package_name: Name of the package
        chunk_content: Content of the chunk to distill
        part_num: Part number (1-based)
        num_parts: Total number of parts
        package_description: Optional description of the package
        doc_type: Type of documentation (general, api, tutorial)
        
    Returns:
        Prompt for the documentation chunk
    """
    # Parameters for template substitution
    params = {
        "package_name": package_name,
        "package_description": package_description or "Python package",
        "num_parts": num_parts,
        "part_num": part_num,
        "chunk_content": chunk_content,
        "prev_parts_count": num_parts
    }
    
    # Determine which template to use
    if part_num > num_parts:
        # Final part for anything missed
        return FINAL_PROMPT_TEMPLATE.substitute(params)
    
    if part_num == 1:
        # First part - use specialized template based on doc_type
        if doc_type == "api":
            return API_REFERENCE_TEMPLATE.substitute(params)
        elif doc_type == "tutorial":
            return TUTORIAL_TEMPLATE.substitute(params)
        else:
            return MAIN_PROMPT_TEMPLATE.substitute(params)
    else:
        # Follow-up part
        return FOLLOW_UP_PROMPT_TEMPLATE.substitute(params)
