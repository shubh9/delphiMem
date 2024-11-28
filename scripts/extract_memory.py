from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime
import inquirer
from typing import Dict, List
import asyncio
from tqdm import tqdm
from memory_extractors.base_point_extractor import BasePointExtractor
from memory_extractors.labeled_point_extractor import LabeledPointExtractor
from memory_extractors.structured_point_extractor import StructuredPointExtractor

load_dotenv()

def get_extractor_class():
    """Present user with a selection of available extractors"""
    extractors = [
        ("Structured Point Extractor", StructuredPointExtractor),
        ("Labeled Point Extractor", LabeledPointExtractor),
        ("Base Point Extractor", BasePointExtractor),

        # Add more extractors here as they become available
    ]
    
    questions = [
        inquirer.List('extractor',
                     message="Select the memory extraction method",
                     choices=[(name, cls) for name, cls in extractors])
    ]
    
    answers = inquirer.prompt(questions)
    return answers['extractor']

async def process_person(person_id: int, messages: List[dict], extractor, pbar) -> dict:
    """Process a single person's messages asynchronously"""
    try:
        print(f"Starting processing for person {person_id}")
        person_memories = extractor.extract_memories(messages, person_id)
        if person_memories:
            memory_entry = {
                'person_id': person_id,
                'extracted_memories': person_memories
            }
            pbar.update(1)
            print(f"Finished processing person {person_id}")
            return memory_entry
    except Exception as e:
        print(f'Error processing person {person_id}: {str(e)}')
    return None

async def extract_memories_from_conversations(extractor_class=None):
    """Extract memories from saved conversations using the specified extractor"""
    try:
        if extractor_class is None:
            extractor_class = get_extractor_class()
            
        data_dir = Path(__file__).parent.parent / "data"
        conversations_file = data_dir / "fake_conversations.json"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        extractor_name = extractor_class.__name__.lower()
        output_file = data_dir / "extracted_memories" / f"{extractor_name}_{timestamp}.json"
        
        if not conversations_file.exists():
            print("No conversations found to analyze")
            return

        with open(conversations_file, 'r') as f:
            conversations = json.load(f)

        # Group conversations by person_id
        conversations_by_person = {}
        for convo in conversations:
            person_id = convo['person_id']
            if person_id in [1, 2, 3]:  # Skip specific persons
                continue
            if person_id not in conversations_by_person:
                conversations_by_person[person_id] = []
            conversations_by_person[person_id].extend(convo['messages'])

        extractor = extractor_class()
        tasks = []
        
        # Create progress bar
        pbar = tqdm(total=len(conversations_by_person), desc="Processing people")
        
        # Create tasks for each person
        for person_id, messages in conversations_by_person.items():
            # Create coroutine and add to tasks list
            task = asyncio.create_task(process_person(person_id, messages, extractor, pbar))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Filter out None results and save
        extracted_memories = [r for r in results if r is not None]
        
        with open(output_file, 'w') as f:
            json.dump(extracted_memories, f, indent=2)
            
        pbar.close()
        print(f"\nTotal memories extracted: {len(extracted_memories)}")
        print(f"Extracted memories saved to {output_file}")
            
    except KeyboardInterrupt:
        print("\nExtraction interrupted by user")
        return
    except Exception as e:
        print(f'Error extracting memories: {str(e)}')

if __name__ == "__main__":
    asyncio.run(extract_memories_from_conversations()) 