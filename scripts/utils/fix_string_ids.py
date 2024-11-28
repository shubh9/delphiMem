import json
from pathlib import Path
import sys

def load_json_file(filepath: str):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def save_json_file(filepath: str, data):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {filepath}")
    except Exception as e:
        print(f"Error saving {filepath}: {e}")

def convert_id_to_int(value):
    """Convert ID to integer if it's a string number"""
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value

def fix_mock_people(data):
    """Fix IDs in mock_people.json"""
    changes = 0
    for person in data:
        for fact in person['facts']:
            old_id = fact['id']
            fact['id'] = convert_id_to_int(fact['id'])
            if old_id != fact['id']:
                changes += 1
                print(f"Converted fact ID from {old_id} to {fact['id']}")
    return data, changes

def fix_memory_quiz(data):
    """Fix IDs in memory_quiz.json"""
    changes = 0
    for person in data:
        for question in person['questions']:
            if 'right_memory_ids' in question:
                old_ids = question['right_memory_ids']
                question['right_memory_ids'] = [convert_id_to_int(id_) for id_ in question['right_memory_ids']]
                if old_ids != question['right_memory_ids']:
                    changes += 1
                    print(f"Converted right_memory_ids from {old_ids} to {question['right_memory_ids']}")
            
            # Also convert the question ID if it exists
            if 'id' in question:
                old_id = question['id']
                question['id'] = convert_id_to_int(question['id'])
                if old_id != question['id']:
                    changes += 1
                    print(f"Converted question ID from {old_id} to {question['id']}")
                    
    return data, changes

def fix_extracted_memories(data):
    """Fix IDs in extracted_memories.json"""
    changes = 0
    for person in data:
        for memory in person['extracted_memories']:
            if 'id' in memory:
                if isinstance(memory['id'], list):
                    old_ids = memory['id']
                    memory['id'] = [convert_id_to_int(id_) for id_ in memory['id']]
                    if old_ids != memory['id']:
                        changes += 1
                        print(f"Converted memory IDs from {old_ids} to {memory['id']}")
                else:
                    old_id = memory['id']
                    memory['id'] = convert_id_to_int(memory['id'])
                    if old_id != memory['id']:
                        changes += 1
                        print(f"Converted memory ID from {old_id} to {memory['id']}")
    return data, changes

def fix_rag_mapping(data):
    """Fix IDs in BasicRAG_pre_fixed_mapping.json"""
    changes = 0
    for person in data:
        for question in person['questions']:
            # Fix predicted_memory_ids
            if 'predicted_memory_ids' in question:
                old_ids = question['predicted_memory_ids']
                question['predicted_memory_ids'] = [convert_id_to_int(id_) for id_ in question['predicted_memory_ids']]
                if old_ids != question['predicted_memory_ids']:
                    changes += 1
                    print(f"Converted predicted_memory_ids from {old_ids} to {question['predicted_memory_ids']}")
            
            # Fix actual_memory_ids
            if 'actual_memory_ids' in question:
                old_ids = question['actual_memory_ids']
                question['actual_memory_ids'] = [convert_id_to_int(id_) for id_ in question['actual_memory_ids']]
                if old_ids != question['actual_memory_ids']:
                    changes += 1
                    print(f"Converted actual_memory_ids from {old_ids} to {question['actual_memory_ids']}")
                    
    return data, changes

def main():
    # Get project root directory
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "data"
    tests_dir = data_dir / "tests"
    
    # Files to process
    files_to_process = {
        # Main data files
        str(data_dir / "mock_people.json"): fix_mock_people,
        str(data_dir / "memory_quiz.json"): fix_memory_quiz,
        str(data_dir / "extracted_memories.json"): fix_extracted_memories,
        
        # Test files
        str(tests_dir / "BasicRAG_pre_fixed_mapping.json"): fix_rag_mapping
    }
    
    total_changes = 0
    
    # Process each file
    for filepath_str, fix_function in files_to_process.items():
        filepath = Path(filepath_str)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            continue
            
        print(f"\nProcessing {filepath.name}...")
        data = load_json_file(filepath)
        if data is None:
            continue
            
        updated_data, changes = fix_function(data)
        if changes > 0:
            save_json_file(filepath, updated_data)
            print(f"Made {changes} changes in {filepath.name}")
            total_changes += changes
        else:
            print(f"No changes needed in {filepath.name}")
    
    print(f"\nTotal changes across all files: {total_changes}")

if __name__ == "__main__":
    main() 