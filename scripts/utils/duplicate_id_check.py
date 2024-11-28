import json
from collections import defaultdict

def generate_new_id(existing_ids):
    new_id = max(int(id_) for id_ in existing_ids) + 1
    return str(new_id)

def mock_people_fix_duplicate_ids(json_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    id_counts = defaultdict(list)
    all_ids = set()
    
    # First pass to collect all IDs and duplicates
    for person in data:
        for fact in person["facts"]:
            fact_id = fact["id"]
            all_ids.add(fact_id)
            id_counts[fact_id].append((person["person_id"], fact))
    
    # Second pass to fix duplicates
    changes_made = {}
    for fact_id, occurrences in id_counts.items():
        if len(occurrences) > 1:
            # Skip first occurrence (keep original ID)
            for person_id, fact in occurrences[1:]:
                new_id = generate_new_id(all_ids)
                all_ids.add(new_id)
                changes_made[fact_id] = changes_made.get(fact_id, []) + [(person_id, fact["content"], new_id)]
                fact["id"] = new_id
    
    # Save modified data
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=2)
    
    return changes_made

def extracted_memories_fix_duplicate_ids(json_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    content_counts = defaultdict(list)
    
    # Collect all contents and their occurrences
    for person in data:
        for memory in person["extracted_memories"]:
            # Use content as the key instead of ID
            memory_content = memory["content"]
            content_counts[memory_content].append((person["person_id"], memory["id"]))
    
    # Find and return duplicates
    duplicates = {content: occurrences for content, occurrences in content_counts.items() 
                 if len(occurrences) > 1}
    
    return duplicates

def find_missing_ids(mock_people_path, extracted_memories_path):
    # Read both JSON files
    with open(mock_people_path, 'r') as file:
        mock_data = json.load(file)
    with open(extracted_memories_path, 'r') as file:
        extracted_data = json.load(file)
    
    # Get all IDs and content from mock_people
    mock_ids = {}  # Changed to dict to store id -> (person_id, content)
    for person in mock_data:
        for fact in person["facts"]:
            # Store ID as is, since they are all numbers
            mock_ids[fact["id"]] = (person["person_id"], fact["content"])
    
    # Get all IDs from extracted_memories
    extracted_ids = set()
    for person in extracted_data:
        for memory in person["extracted_memories"]:
            # Store ID as is, since they are all numbers
            for id_num in memory["id"]:
                extracted_ids.add(id_num)
    
    # Find IDs in mock_people that aren't in extracted_memories
    missing_ids = {id_: info for id_, info in mock_ids.items() if id_ not in extracted_ids}
    
    return {
        'missing_ids': missing_ids,
        'total_mock_ids': len(mock_ids),
        'total_extracted_ids': len(extracted_ids)
    }

def print_conversation_message_counts(json_file_path):
    """
    Reads the fake conversations JSON file and prints the number of messages in each conversation.
    
    Args:
        json_file_path (str): Path to the fake conversations JSON file
    """
    try:
        with open(json_file_path, 'r') as file:
            conversations = json.load(file)
        
        print("\nMessage counts in conversations:")
        total_messages = 0
        for idx, conversation in enumerate(conversations, 1):
            message_count = len(conversation['messages'])
            total_messages += message_count
            print(f"Conversation {idx}: {message_count} messages")
        
        print(f"\nTotal conversations: {len(conversations)}")
        print(f"Total messages across all conversations: {total_messages}")
        print(f"Average messages per conversation: {total_messages / len(conversations):.1f}")
            
    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")

# Example usage
if __name__ == "__main__":
    # First check and fix mock_people.json
    changes = mock_people_fix_duplicate_ids("data/mock_people.json")
    if changes:
        print("Fixed duplicate IDs in mock_people.json:")
        for old_id, modifications in changes.items():
            print(f"\nOriginal ID {old_id} had duplicates:")
            for person_id, content, new_id in modifications:
                print(f"  Person {person_id}: '{content}' -> New ID: {new_id}")
    else:
        print("No duplicate IDs found in mock_people.json")

    # Then check extracted_memories.json
    print("\nChecking extracted_memories.json for duplicates...")
    duplicates = extracted_memories_fix_duplicate_ids("data/extracted_memories.json")
    if duplicates:
        print("\nFound duplicate IDs in extracted_memories.json:")
        for dup_id, occurrences in duplicates.items():
            print(f"\nID {dup_id} appears in multiple entries:")
            for person_id, content in occurrences:
                print(f"  Person {person_id}: '{content}'")
    else:
        print("No duplicate IDs found in extracted_memories.json")

    print("\nChecking for IDs in mock_people.json that don't exist in extracted_memories.json...")
    result = find_missing_ids("data/mock_people.json", "data/extracted_memories.json")
    missing_ids = result['missing_ids']
    
    print(f"\nTotal IDs in mock_people.json: {result['total_mock_ids']}")
    print(f"Total IDs in extracted_memories.json: {result['total_extracted_ids']}")
    
    if missing_ids:
        total_missing = len(missing_ids)
        print(f"\nFound {total_missing} IDs in mock_people.json that don't exist in extracted_memories.json:")
        for id_, (person_id, content) in sorted(missing_ids.items(), key=lambda item: item[1][0]):
            print(f"  ID {id_} (Person {person_id}): '{content}'")
    else:
        print("All IDs in mock_people.json exist in extracted_memories.json")

    print("\nAnalyzing conversation message counts...")
    print_conversation_message_counts("data/fake_conversations.json")