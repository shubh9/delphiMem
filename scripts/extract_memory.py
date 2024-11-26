from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime
import inquirer
from typing import Dict, List
from memory_extractors.base_point_extractor import BasePointExtractor
from memory_extractors.labeled_point_extractor import LabeledPointExtractor
import openai

load_dotenv()

def get_extractor_class():
    """Present user with a selection of available extractors"""
    extractors = [
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

def extract_memories_from_conversations(extractor_class=None):
    """Extract memories from saved conversations using the specified extractor"""
    try:
        if extractor_class is None:
            extractor_class = get_extractor_class()
            
        data_dir = Path(__file__).parent.parent / "data"
        conversations_file = data_dir / "fake_conversations.json"
        
        # Generate output filename with date and extractor name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        extractor_name = extractor_class.__name__.lower()
        output_file = data_dir / f"extracted_memories_{extractor_name}_{timestamp}.json"
        
        if not conversations_file.exists():
            print("No conversations found to analyze")
            return

        with open(conversations_file, 'r') as f:
            conversations = json.load(f)

        # Group conversations by person_id
        conversations_by_person = {}
        for convo in conversations:
            person_id = convo['person_id']
            if person_id not in conversations_by_person:
                conversations_by_person[person_id] = []
            conversations_by_person[person_id].extend(convo['messages'])

        extractor = extractor_class()
        extracted_memories = []
        
        # Process each person's conversations
        for person_id, messages in conversations_by_person.items():
            print(f"Processing person {person_id}")
            
            person_memories = extractor.extract_memories(messages, person_id)
            
            if person_memories:
                memory_entry = {
                    'person_id': person_id,
                    'extracted_memories': person_memories
                }
                extracted_memories.append(memory_entry)
                
                # Save after each person is processed
                with open(output_file, 'w') as f:
                    json.dump(extracted_memories, f, indent=2)
                print(f"Saved memories for person {person_id}")
            
        print(f"Total memories extracted: {len(extracted_memories)}")
        print(f"Extracted memories saved to {output_file}")
            
    except KeyboardInterrupt:
        print("\nExtraction interrupted by user")
        return
    except Exception as e:
        print(f'Error extracting memories: {str(e)}')

if __name__ == "__main__":
    extract_memories_from_conversations() 