import json
import random
import os
from typing import Dict, List, Optional, Union
from openai import AzureOpenAI
import time
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def load_json_file(filepath: str) -> List[dict]:
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json_file(filepath: str, data: List[dict]):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def get_matching_id_from_gpt(memory: str, facts: List[dict]) -> Optional[int]:
    """Use GPT to find a matching fact ID for a given memory."""
    
    prompt = f"""Given this labeled memory: "{memory}"

And these facts (with IDs):
{json.dumps(facts, indent=2)}

Compare the semantic meaning, ignoring the label format differences. For example, "Bob<user> lives in Seattle<city>" matches "Lives in Seattle".

If the memory matches one of the facts, return ONLY the ID number of the matching fact.
If the memory doesn't match any fact, return "NO_MATCH".

Return your answer in this exact format - just the ID number or "NO_MATCH". Nothing else.
"""
    
    print("prompt:\n", prompt)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user", 
                "content": prompt
            }]
        )
        
        result = response.choices[0].message.content.strip()
        
        if result == "NO_MATCH":
            return None
            
        return int(result)
    
    except Exception as e:
        print(f"Error getting GPT response for memory: {memory}")
        print(f"Error: {e}")
        raise e

def check_unmatched_facts(unmatched_facts: List[dict], memories: List[dict]) -> Dict[int, List[int]]:
    """Check if any unmatched facts should be matched to existing memories."""
    
    prompt_template = """Given this unmatched fact:
{fact}

And these labeled memories:
{memories}

Compare the semantic meaning, ignoring the label format differences. For example, "Bob<user> lives in Seattle<city>" matches "Lives in Seattle".

Should this fact be matched with any of the memories? If yes, return the memory ID(s) as a comma-separated list.
If no match, return "NO_MATCH".

Return ONLY the ID(s) or "NO_MATCH". Example: "1234,5678" or "NO_MATCH"
"""

    matches = {}
    
    for fact in unmatched_facts:
        print(f"\nChecking fact {fact['id']}")
        
        prompt = prompt_template.format(
            fact=json.dumps(fact, indent=2),
            memories=json.dumps(memories, indent=2)
        )
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            result = response.choices[0].message.content.strip()
            
            if result != "NO_MATCH":
                memory_ids = [int(id_str) for id_str in result.split(',')]
                matches[fact['id']] = memory_ids
                print(f"Matched fact {fact['id']} to memories {memory_ids}")
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"Error checking fact {fact['id']}: {e}")
            raise e
            
    return matches

def generate_unique_id(existing_ids: set) -> int:
    """Generate a unique 4-digit ID that starts with 0 and hasn't been used."""
    while True:
        new_id = random.randint(1000, 9999)  # Generate 4 digits
        new_id = int(f"0{new_id}")  # Add leading 0 to make it 5 digits
        if new_id not in existing_ids:
            return new_id

def main():
    # Update file path to use the labeled point extractor output
    memories_data = load_json_file('data/extracted_memories_labeledpointextractor_20241125_2240.json')
    people_data = load_json_file('data/mock_people.json')
    
    # Get existing IDs and track matched IDs
    existing_ids = {fact['id'] for person in people_data for fact in person['facts']}
    matched_ids = set()
    
    # Process memories - First Pass (single ID match)
    updated_memories = []
    for person in memories_data:
        print(f"Processing person {person}")
        updated_person = {
            'person_id': person['person_id'],
            'extracted_memories': []
        }
        
        for memory in person['extracted_memories']:
            # Convert existing single ID to array if needed
            if isinstance(memory.get('id'), int):
                memory['id'] = [memory['id']]
                
            print(f"Processing memory: {memory['content']}")
            
            memory_id = get_matching_id_from_gpt(
                memory['content'], 
                people_data[person['person_id']-1]['facts']
            )
            
            if memory_id is not None:
                print(f"Found Match Using ID: {memory_id}")
                matched_ids.add(memory_id)
                updated_person['extracted_memories'].append({
                    'id': [memory_id],
                    'content': memory['content']
                })
            else:
                # Store memory without ID for now
                updated_person['extracted_memories'].append({
                    'id': [],
                    'content': memory['content']
                })
            
            time.sleep(0.5)
            
        updated_memories.append(updated_person)
    
    # Find unmatched facts
    unmatched_facts = []
    for person in people_data:
        for fact in person['facts']:
            if fact['id'] not in matched_ids:
                unmatched_facts.append(fact)
    
    print(f"\nFound {len(unmatched_facts)} unmatched facts")
    
    # Second Pass - Check unmatched facts
    if unmatched_facts:
        all_memories = []
        for person in updated_memories:
            all_memories.extend(person['extracted_memories'])
            
        fact_matches = check_unmatched_facts(unmatched_facts, all_memories)
        
        # Update memories with additional matched facts
        for fact_id, memory_ids in fact_matches.items():
            for person in updated_memories:
                for memory in person['extracted_memories']:
                    if any(mem_id in memory_ids for mem_id in memory['id']):
                        if fact_id not in memory['id']:
                            memory['id'].append(fact_id)
                            print(f"Added fact {fact_id} to memory with IDs {memory['id']}")
    
    # Generate new IDs for unmatched memories
    for person in updated_memories:
        for memory in person['extracted_memories']:
            if not memory['id']:  # If no matches were found
                new_id = generate_unique_id(existing_ids)
                memory['id'] = [new_id]
                existing_ids.add(new_id)
                print(f"Generated new ID {new_id} for unmatched memory: {memory['content']}")
    
    # Save updated memories
    save_json_file('data/extracted_memories.json', updated_memories)
    print("Memory matching complete! Updated extracted_memories.json with IDs.")

if __name__ == "__main__":
    main() 