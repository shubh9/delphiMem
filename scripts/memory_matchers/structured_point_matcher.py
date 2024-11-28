from typing import Dict, List, Optional
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import json
import inquirer
from pathlib import Path
import random

class StructuredPointMatcher:
    MATCHING_PROMPT = """
    You are a memory matching system. Your task is to find which fact from a list matches a target memory, if any.
    
    Target Memory:
    {target_memory}
    
    List of Facts to Search:
    {facts_list}
    
    Rules for matching:
    1. Focus on key identifying information and semantic meaning
    2. Consider contextual clues and relationships
    3. Be conservative - only match if you're confident they represent the same information
    4. Ignore formatting differences (e.g. "Lives in Seattle" matches "location: Seattle")
    5. Match based on meaning, not exact wording
    
    Return ONLY the ID of the matching fact, or "NO_MATCH" if no match is found.
    If multiple matches are found, return the best match only.

    ONLY RETURN THE ID, NOTHING ELSE, no other text or explanation or anything else
    
    Example responses:
    "01234"
    "NO_MATCH"
    """

    def __init__(self):
        load_dotenv()
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-15-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.used_fact_ids = set()  # Add this to track used fact IDs

    def _format_facts_list(self, facts: List[Dict]) -> str:
        """Format facts into a readable list for comparison"""
        formatted_facts = []
        for fact in facts:
            formatted_facts.append(f"ID: {fact['id']}\nContent: {fact['content']}")
        return "\n\n".join(formatted_facts)

    def _generate_unique_id(self) -> int:
        """Generate a unique 5-digit ID that starts with 0."""
        while True:
            new_id = random.randint(1000, 9999)
            new_id = int(f"0{new_id}")
            if new_id not in self.used_fact_ids:
                return new_id

    def find_matching_fact(self, memory: Dict, facts: List[Dict]) -> Optional[str]:
        """
        Find a fact that matches a given memory using LLM comparison.
        Returns matching fact ID if found, None otherwise.
        """
        # Filter out already used facts
        available_facts = [fact for fact in facts if fact['id'] not in self.used_fact_ids]
        
        if not available_facts:
            # If no available facts, generate a new random ID
            new_id = self._generate_unique_id()
            self.used_fact_ids.add(new_id)
            return new_id

        memory_text = f"{memory['attribute']}: {memory['content']}"
        facts_text = self._format_facts_list(available_facts)
        
        prompt = self.MATCHING_PROMPT.format(
            target_memory=memory_text,
            facts_list=facts_text
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        result = response.choices[0].message.content.strip()
        
        if "NO_MATCH" in result:
            # Generate new random ID for unmatched memory
            new_id = self._generate_unique_id()
            self.used_fact_ids.add(new_id)
            return new_id
        
        # Validate that result is a 5-digit number
        if not result.isdigit() or len(result) != 5:
            raise ValueError(f"LLM returned invalid fact ID: {result}. Expected a 5-digit number or 'NO_MATCH'")
        
        fact_id = int(result)
        self.used_fact_ids.add(fact_id)  # Mark this fact ID as used
        return fact_id

    def process_memory_file(self, memory_file: Path):
        """Process all memories against facts in mock_people"""
        # Load memories
        with open(memory_file, 'r') as f:
            memories = json.load(f)
            
        # Load mock person data
        with open('data/mock_people.json', 'r') as f:
            mock_people = json.load(f)
            
        # Calculate total values for all extracted memories
        total_values = sum(
            len(values) for person_memory in memories
            for extracted_memory in person_memory['extracted_memories']
            for values in extracted_memory['Profile'].values()
        )
        
        # Process each memory against facts for the corresponding person
        for person_memory in memories:
            person_id = person_memory['person_id']
            print(f"\nProcessing memories for person {person_id}")
            
            # Get facts for this person
            person_facts = next(p['facts'] for p in mock_people if p['person_id'] == person_id)
            
            current_index = 0  # Initialize current index

            # Process each memory in the extracted memories
            for extracted_memory in person_memory['extracted_memories']:
                profile = extracted_memory['Profile']
                for attribute, values in profile.items():
                    for value in values:
                        current_index += 1  # Increment current index
                        
                        # Create memory dict for matching
                        memory_dict = {
                            'attribute': attribute,
                            'content': value['content']
                        }
                        
                        # Find matching fact
                        matching_fact_id = self.find_matching_fact(memory_dict, person_facts)
                        
                        if matching_fact_id:
                            # Update memory with matching fact ID
                            value['mem_id'] = matching_fact_id
                            print(f"Matched memory {current_index} out of {total_values}: '{value['content']}' to fact {matching_fact_id}")
        
        # Save updated memories back to file
        with open(memory_file, 'w') as f:
            json.dump(memories, f, indent=2)
            
        print(f"\nProcessing complete. Updated memories saved to {memory_file}")

    @staticmethod
    def select_memory_file():
        """Let user select a memory file from data/extracted_memories"""
        memories_dir = Path("data/extracted_memories")
        json_files = list(memories_dir.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError("No JSON files found in data/extracted_memories")
        
        questions = [
            inquirer.List('file',
                         message="Select the memory file to process",
                         choices=[f.name for f in json_files])
        ]
        
        answers = inquirer.prompt(questions)
        return memories_dir / answers['file']

def main():
    matcher = StructuredPointMatcher()
    memory_file = matcher.select_memory_file()
    matcher.process_memory_file(memory_file)

if __name__ == "__main__":
    main()