import re
from typing import List, Dict, Optional

def extract_predictive_plan(planner_output: str) -> Optional[List[Dict[str, str]]]:
    """
    Extract plan information from high-level planner output.
    
    Args:
        planner_output: Full output string from high-level planner
        
    Returns:
        List[Dict[str, str]]: List of dictionaries containing task info, each with 'task' and 'status' keys.
        Returns None if no plan was found.
    """
    if not planner_output or not isinstance(planner_output, str):
        return None
    
    # Define regular expression matching XML-format todo list
    # Matches <update_todo_list><todos>...content...</todos></update_todo_list>
    pattern = r'<update_todo_list>\s*<todos>(.*?)</todos>\s*</update_todo_list>'
    match = re.search(pattern, planner_output, re.DOTALL)
    
    if not match:
        # If XML format is not found, try matching task format directly
        return extract_todo_list_from_text(planner_output)
    
    todos_content = match.group(1)
    
    # Extract each line of task, format: [status] task description
    todo_items = []
    lines = todos_content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Match tasks starting with [ ], [-], or [x]
        task_match = re.match(r'^(\[[ x-]\])\s*(.+)$', line)
        if task_match:
            status = task_match.group(1).strip()
            task_description = task_match.group(2).strip()
            
            # Standardize status representation
            if status == '[ ]':
                standardized_status = 'pending'
            elif status == '[-]':
                standardized_status = 'in_progress'
            elif status == '[x]':
                standardized_status = 'completed'
            else:
                standardized_status = 'unknown'
                
            if task_description:  # Only add tasks that have a description
                todo_items.append({
                    'task': task_description,
                    'status': standardized_status
                })
    
    return todo_items if todo_items else None


def extract_todo_list_from_text(text: str) -> Optional[List[Dict[str, str]]]:
    """
    Directly extract tasks in todo list format from plain text.
    
    Args:
        text: Text containing todo list
        
    Returns:
        List[Dict[str, str]]: List of dictionaries containing task info
    """
    if not text or not isinstance(text, str):
        return None
    
    # Match tasks starting with [ ], [-], or [x], across multiple lines
    todo_items = []
    
    # Split text by line
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Match [status] task description format
        task_match = re.match(r'^(\[[ x-]\])\s*(.+)$', line)
        if task_match:
            status = task_match.group(1).strip()
            task_description = task_match.group(2).strip()
            
            # Standardize status representation
            if status == '[ ]':
                standardized_status = 'pending'
            elif status == '[-]':
                standardized_status = 'in_progress'
            elif status == '[x]':
                standardized_status = 'completed'
            else:
                standardized_status = 'unknown'
                
            if task_description:  # Only add tasks that have a description
                todo_items.append({
                    'task': task_description,
                    'status': standardized_status
                })
    
    return todo_items if todo_items else None


def get_tasks_by_status(todo_list: List[Dict[str, str]], status: str) -> List[Dict[str, str]]:
    if not todo_list:
        return []
    
    return [task for task in todo_list if task.get('status') == status]


def get_pending_tasks(todo_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return get_tasks_by_status(todo_list, 'pending')


def get_in_progress_tasks(todo_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return get_tasks_by_status(todo_list, 'in_progress')


def get_completed_tasks(todo_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return get_tasks_by_status(todo_list, 'completed')


def print_todo_list(todo_list: List[Dict[str, str]]) -> None:
    if not todo_list:
        print("No tasks found")
        return
    
    print("Todo List:")
    for i, task in enumerate(todo_list):
        status = task.get('status', 'unknown')
        task_desc = task.get('task', '')
        
        status_symbol = {
            'pending': '[ ]',
            'in_progress': '[-]',
            'completed': '[x]'
        }.get(status, '[?]')
        
        print(f"  {status_symbol} {task_desc}")
