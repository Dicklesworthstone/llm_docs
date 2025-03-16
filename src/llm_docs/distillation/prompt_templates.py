"""
Improved prompt templates for documentation distillation.
"""

from string import Template

# Base templates for different parts of the distillation process
MAIN_PROMPT_TEMPLATE = Template("""
I attached the COMPLETE documentation for ${package_name}, the python library ${package_description}.

The documentation contains a lot of "fluff" and references to things which we do NOT care about. I pasted the docs manually from the documentation website.  

Your goal is to meticulously read and understand everything and then distill it down into the most compact and clear form that would still be easily intelligible to YOU in a new conversation where you had never seen the original documentation text. It's too long to do in one shot, so let's do it in ${num_parts} sections, starting now with the first part. 

Feel free to rearrange the order of sections so that it makes most sense to YOU since YOU will be the one consuming the output! Don't worry if it would be confusing or hard to follow by a human, it's not for a human, it's for YOU to quickly and efficiently convey all the relevant information in the documentation.

Focus on:
1. Core functionality and API details
2. Key classes, methods, and functions with their parameters
3. Important usage patterns and examples
4. Configuration options and important defaults
5. Common error cases and how to handle them

Strip out:
1. Marketing language and fluff
2. Redundant examples
3. Verbose explanations that don't add technical information
4. Information about services or features not relevant to the library itself

${chunk_content}
""")

FOLLOW_UP_PROMPT_TEMPLATE = Template("""
continue with part ${part_num} of ${num_parts}

Remember to maintain the same style and formatting as in previous parts. Continue distilling the documentation into the most compact and clear form for an AI assistant to understand quickly.

${chunk_content}
""")

FINAL_PROMPT_TEMPLATE = Template("""
ok now make a part ${part_num} with all the important stuff that you left out from the full docs in the ${prev_parts_count} parts you already wrote above

Focus specifically on:
1. Any important API methods, classes or functions that may have been missed
2. Edge cases and important warnings
3. Advanced configuration options
4. Performance considerations
5. Compatibility information

This is your chance to add anything that was left out but would be important for using the library effectively.
""")

# Specialized templates for different types of documentation
API_REFERENCE_TEMPLATE = Template("""
I attached the API reference documentation for ${package_name}, the python library ${package_description}.

Focus specifically on summarizing:
1. Function signatures, parameters, return values, and exceptions
2. Class hierarchies and their methods
3. Important constants and configuration options
4. Key usage patterns and constraints

Strip away explanatory prose in favor of concise, structured descriptions optimized for an AI assistant to reference quickly. It's too long to do in one shot, so let's do it in ${num_parts} sections, starting now with the first part.

Present the information in a format that would be most efficient for an AI to understand and recall when helping users with this library. Group related functionality together logically, even if they appear in different parts of the original documentation.

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

Extract the essence of how the library is meant to be used, with just enough example code to illustrate the patterns.

${chunk_content}
""")

QUICK_REFERENCE_TEMPLATE = Template("""
I attached the documentation for ${package_name}, the python library ${package_description}.

Create an extremely concise quick reference guide focused on:
1. Core API components with minimal examples
2. Key function signatures and class methods
3. Essential configuration parameters
4. Common usage patterns

This should be in a format optimized for an AI assistant to quickly look up information while helping users. Use a consistent structure with clear hierarchical organization. It's too long to do in one shot, so let's do it in ${num_parts} sections, starting now with the first part.

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
        doc_type: Type of documentation (general, api, tutorial, quick_reference)
        
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
        elif doc_type == "quick_reference":
            return QUICK_REFERENCE_TEMPLATE.substitute(params)
        else:
            return MAIN_PROMPT_TEMPLATE.substitute(params)
    else:
        # Follow-up part
        return FOLLOW_UP_PROMPT_TEMPLATE.substitute(params)
