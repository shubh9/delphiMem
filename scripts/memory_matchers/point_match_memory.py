import json
import random
import os
from typing import Dict, List, Optional, Union, Tuple
from openai import AzureOpenAI
import time
from dotenv import load_dotenv
import glob
import inquirer
import datetime
from pathlib import Path
from tqdm import tqdm
import asyncio
import aiohttp

# Load environment variables
load_dotenv()

# Global constants
DATA_DIR = "data"
MEMORIES_DIR = os.path.join(DATA_DIR, "extracted_memories")
MOCK_PEOPLE_PATH = os.path.join(DATA_DIR, "mock_people.json")

# Initialize OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Add these new classes at the top level, after the imports

class MemoryFormat:
    STRUCTURED = "structured"
    FLAT = "flat"

class MemoryConverter:
    @staticmethod
    def to_flat_memories(structured_data: dict) -> List[dict]:
        """Convert structured profile data to flat memory format."""
        flat_memories = []
        
        for person_data in structured_data:
            person_memories = []
            extracted_memories = person_data['extracted_memories']
            
            for person_info in extracted_memories:
                profile = person_info.get('Profile', {})
                
                # Convert each profile section to flat memories
                for category, items in profile.items():
                    if isinstance(items, list):
                        for item in items:
                            memory = {
                                'content': f"{category}: {item.get('content')}",
                                'id': item.get('mem_id', [])
                            }
                            person_memories.append(memory)
            
            flat_memories.append({
                'person_id': person_data['person_id'],
                'extracted_memories': person_memories
            })
        return flat_memories

    @staticmethod
    def to_structured_memories(flat_memories: List[dict], original_structure: List[dict]) -> List[dict]:
        """Convert flat memories back to structured format."""
        result = []
        
        for person_data in original_structure:
            updated_person = {
                'person_id': person_data['person_id'],
                'extracted_memories': []
            }
            
            # Get the original structure
            original_extracted = person_data['extracted_memories']
            
            # For each person in the original structure
            for person_info in original_extracted:
                # print("person_info:\n", person_info)
                person_id = person_info['Id']
                updated_memory = {
                    'Id': person_id,
                    'Description': person_info.get('Description', ''),
                    'Profile': {},
                    'Connections': person_info.get('Connections', [])
                }
                
                # Initialize profile categories from original
                for category in person_info.get('Profile', {}).keys():
                    updated_memory['Profile'][category] = []
                
                updated_person['extracted_memories'].append(updated_memory)
                
            # Find matching flat memories and update the structure
            matching_flat = next(p for p in flat_memories if p['person_id'] == person_data['person_id'])
            for memory in matching_flat['extracted_memories']:
                category, content = memory['content'].split(':', 1)
                category = category.strip().lower()
                content = content.strip()
                
                # Find the right person and category
                for person_info in updated_person['extracted_memories']:
                    profile = person_info['Profile']
                    if category in profile:
                        mem_entry = {
                            'content': content,
                            'mem_id': memory['id']
                        }
                        profile[category].append(mem_entry)
            
            result.append(updated_person)
        
        return result

# Add this new class to handle async API calls
class AsyncOpenAIClient:
    def __init__(self, api_key: str, endpoint: str, api_version: str):
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }

    async def create_chat_completion(self, session: aiohttp.ClientSession, messages: List[dict]) -> str:
        """Make async API call to Azure OpenAI."""
        url = f"{self.endpoint}/openai/deployments/gpt-4o/chat/completions?api-version={self.api_version}"
        
        async with session.post(url, json={"messages": messages}, headers=self.headers) as response:
            if response.status == 200:
                data = await response.json()
                return data['choices'][0]['message']['content'].strip()
            else:
                raise Exception(f"API call failed with status {response.status}")

# Update the MemoryMatcher class
class MemoryMatcher:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.original_data = self._load_json_file(input_file)
        self.format = self._detect_format(self.original_data)
        print(f"Detected format: {self.format}")
        
        if self.format == MemoryFormat.STRUCTURED:
            self.memories_data = MemoryConverter.to_flat_memories(self.original_data)
        else:
            self.memories_data = self.original_data
            
        self.people_data = self._load_json_file(MOCK_PEOPLE_PATH)
        self.existing_ids = {fact['id'] for person in self.people_data for fact in person['facts']}
        self.matched_ids = set()
        self.updated_memories = []

    def _detect_format(self, data: List[dict]) -> str:
        """Detect if the data is in structured or flat format."""
        if (len(data) > 0 and
            'extracted_memories' in data[0] and
            isinstance(data[0]['extracted_memories'], list) and
            len(data[0]['extracted_memories']) > 0):
            for memory in data[0]['extracted_memories']:
                if 'Description' in memory:
                    return MemoryFormat.STRUCTURED
        return MemoryFormat.FLAT

    @staticmethod
    def _load_json_file(filepath: str) -> List[dict]:
        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def _save_json_file(filepath: str, data: List[dict]):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _generate_unique_id(self) -> int:
        """Generate a unique 5-digit ID that starts with 0."""
        while True:
            new_id = random.randint(1000, 9999)
            new_id = int(f"0{new_id}")
            if new_id not in self.existing_ids:
                return new_id

    def _get_output_path(self, prefix: str) -> str:
        """Generate output filepath with given prefix."""
        filename = f"{prefix}_{os.path.splitext(os.path.basename(self.input_file))[0]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json"
        return os.path.join(MEMORIES_DIR, filename)

    def _process_memory(self, memory: dict, person_facts: List[dict]) -> dict:
        """Process a single memory synchronously."""
        if isinstance(memory.get('id'), int):
            memory['id'] = [memory['id']]

        if memory.get('id'):
            self.matched_ids.update(memory['id'])
            return memory

        prompt = f"""Given this labeled memory: "{memory['content']}"

And these facts (with IDs):
{json.dumps(person_facts, indent=2)}

Compare the semantic meaning, ignoring the label format differences. For example, "Bob<user> lives in Seattle<city>" matches "Lives in Seattle".
It should be a near perfect match however, if the memory doesn't match any fact, return "NO_MATCH".

If the memory matches one of the facts, return ONLY the ID number of the matching fact.
If the memory doesn't match any fact, return "NO_MATCH".

Return your answer in this exact format - just the ID number or "NO_MATCH". Nothing else.
"""

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
                return {'id': [], 'content': memory['content']}
                
            memory_id = int(result)
            self.matched_ids.add(memory_id)
            return {'id': [memory_id], 'content': memory['content']}
            
        except Exception as e:
            print(f"Error processing memory: {memory['content']}")
            print(f"Error: {e}")
            raise e

    def first_pass(self):
        """Perform first pass of memory matching synchronously."""
        total_memories = sum(len(person['extracted_memories']) for person in self.memories_data)
        progress_bar = tqdm(total=total_memories, desc="Processing memories")
        
        results = {}
        for person in self.memories_data:
            person_facts = self.people_data[person['person_id']-1]['facts']
            results[person['person_id']] = []
            
            for memory in person['extracted_memories']:
                result = self._process_memory(memory, person_facts)
                results[person['person_id']].append(result)
                progress_bar.update(1)
        
        progress_bar.close()
        
        # Organize results back into the original structure
        self.updated_memories = []
        for person in self.memories_data:
            updated_person = {
                'person_id': person['person_id'],
                'extracted_memories': results[person['person_id']]
            }
            self.updated_memories.append(updated_person)

        # Convert back to structured format if needed
        if self.format == MemoryFormat.STRUCTURED:
            save_data = MemoryConverter.to_structured_memories(
                self.updated_memories, 
                self.original_data
            )
        else:
            save_data = self.updated_memories

        # Save intermediate results
        self._save_json_file(self.input_file, save_data)
        print(f"\nFirst pass results saved to: {self.input_file}")

    def second_pass(self):
        """Perform second pass to match unmatched facts."""
        # Filter people_data to only include person_ids present in memories_data
        valid_person_ids = {person['person_id'] for person in self.memories_data}
        
        unmatched_facts = [
            fact for person in self.people_data 
            if person['person_id'] in valid_person_ids  # Check if person_id is in memories_data
            for fact in person['facts'] 
            if fact['id'] not in self.matched_ids
        ]
        
        print(f"\nFound {len(unmatched_facts)} unmatched facts")
        
        if unmatched_facts:
            all_memories = [
                memory for person in self.updated_memories 
                for memory in person['extracted_memories']
            ]
            
            # Add progress bar for unmatched facts
            with tqdm(total=len(unmatched_facts), desc="Processing unmatched facts") as pbar:
                fact_matches = {}
                for fact in unmatched_facts:
                    matches = check_unmatched_facts([fact], all_memories)
                    fact_matches.update(matches)
                    pbar.update(1)
            
            # Update memories with additional matched facts
            print("\nUpdating memories with matched facts...")
            for fact_id, memory_ids in fact_matches.items():
                for person in self.updated_memories:
                    for memory in person['extracted_memories']:
                        if any(mem_id in memory_ids for mem_id in memory['id']):
                            if fact_id not in memory['id']:
                                memory['id'].append(fact_id)
                                print(f"Added fact {fact_id} to memory with IDs {memory['id']}")

    def generate_new_ids(self):
        """Generate new IDs for unmatched memories."""
        for person in self.updated_memories:
            for memory in person['extracted_memories']:
                if not memory['id']:
                    new_id = self._generate_unique_id()
                    memory['id'] = [new_id]
                    self.existing_ids.add(new_id)
                    print(f"Generated new ID {new_id} for unmatched memory: {memory['content']}")

    def save_results(self):
        """Save final results back to the original file."""
        if self.format == MemoryFormat.STRUCTURED:
            final_data = MemoryConverter.to_structured_memories(
                self.updated_memories, 
                self.original_data
            )
        else:
            final_data = self.updated_memories
            
        self._save_json_file(self.input_file, final_data)
        print(f"\nMemory matching complete! Results saved to: {self.input_file}")


# Keep these functions outside the class as they're independent utilities
def get_matching_id_from_gpt(memory: str, facts: List[dict]) -> Optional[int]:
    """Use GPT to find a matching fact ID for a given memory."""
    
    prompt = f"""Given this labeled memory: "{memory}"

And these facts (with IDs):
{json.dumps(facts, indent=2)}

Compare the semantic meaning, ignoring the label format differences. For example, "Bob<user> lives in Seattle<city>" matches "Lives in Seattle".
It should be a near perfect match however, if the memory doesn't match any fact, return "NO_MATCH".

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
    Memories should be very closely related to match, don't match if they are not almost identical.
    If no match, return nothing.

    Return ONLY the ID(s) or "". Example: "1234,5678" or ""
    """

    matches = {}
    
    for fact in unmatched_facts:
        print(f"\nChecking fact {fact['id']}")
        
        prompt = prompt_template.format(
            fact=json.dumps(fact, indent=2),
            memories=json.dumps(memories, indent=2)
        )

        print("Fact being checked:\n", fact)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            result = response.choices[0].message.content.strip()
            print("result|", result, "|")
            
            # Only process non-empty results
            if result and result != '""':  # Add check for quoted empty string
                try:
                    memory_ids = [int(id_str) for id_str in result.split(',')]
                    matches[fact['id']] = memory_ids
                    print(f"Matched fact {fact['id']} to memories {memory_ids}")
                except ValueError as e:
                    print(f"Warning: Could not parse memory IDs from result: {result}")
                    continue
                        
        except Exception as e:
            print(f"Error checking fact {fact['id']}: {e}")
            raise e
            
    return matches

def list_extracted_memories_files() -> List[str]:
    """List all JSON files in the extracted memories directory."""
    return sorted(glob.glob(os.path.join(MEMORIES_DIR, "*.json")))

def pick_memories_file() -> str:
    """Display an inquirer picker interface for selecting the memories file."""
    files = list_extracted_memories_files()
    
    if not files:
        raise ValueError(f"No JSON files found in {MEMORIES_DIR}")
    
    questions = [
        inquirer.List('file',
                     message="Select the extracted memories file to use",
                     choices=[os.path.basename(f) for f in files],
                     carousel=True)
    ]
    
    answers = inquirer.prompt(questions)
    
    if not answers:
        raise KeyboardInterrupt("User cancelled file selection")
        
    selected_basename = answers['file']
    selected_file = next(f for f in files if os.path.basename(f) == selected_basename)
    
    return selected_file

# Update the main function
def main():
    input_file = pick_memories_file()
    matcher = MemoryMatcher(input_file)
    
    # Execute matching process
    matcher.first_pass()
    matcher.generate_new_ids()
    matcher.save_results()

if __name__ == "__main__":
    main() 