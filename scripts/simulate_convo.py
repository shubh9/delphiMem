import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import time

# Add the server directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from simulate_ai_service import create_simulated_chat

def load_mock_people():
    with open('data/mock_people.json', 'r') as file:
        return json.load(file)

def save_conversation(messages: List[Dict], person_id: int):
    """Save conversation to fake_conversations.json"""
    conversation_data = {
        "person_id": person_id,
        "messages": messages
    }
    
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / "fake_conversations.json"
    
    # Load existing conversations or create new list
    if file_path.exists():
        with open(file_path, 'r') as f:
            try:
                conversations = json.load(f)
            except json.JSONDecodeError:
                conversations = []
    else:
        conversations = []
    
    conversations.append(conversation_data)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    
    print(f"\nConversation saved to: {file_path}")

def run_simulation(person_ids: List[int]):
    # Load person data
    mock_people = load_mock_people()
    
    for person_id in person_ids:
        print(f"\n{'='*50}")
        print(f"Starting simulation for person {person_id}")
        print(f"{'='*50}\n")
        
        # Get person facts for current ID
        try:
            person_facts = mock_people[person_id-1]['facts']
        except IndexError:
            print(f"Person ID {person_id} not found in mock_people.json. Skipping...")
            continue
        
        messages = []
        completed_facts = []  # Will store all completed fact IDs
        
        try:
            # Run conversation turns
            for turn in range(40):
                # Log remaining facts every 10 messages
                remaining_facts = [f for f in person_facts if f['id'] not in completed_facts]
                print("\nREMAINING FACTS:")
                print("-"*50)
                for fact in remaining_facts:
                    print(f"- {fact['content']}")
                print("-"*50)

                # If all facts are completed, end the conversation
                if not remaining_facts:
                    print("\nAll facts have been covered! Ending conversation.")
                    print(f"\nTotal messages exchanged: {len(messages)}")
                    break
                
                # If we've reached 40 turns and facts remain, log and stop
                if turn >= 39 and remaining_facts:
                    print("\nReached 40 turns with uncompleted facts. Ending conversation.")
                    print("Uncompleted facts:")
                    for fact in remaining_facts:
                        print(f"- {fact['content']}")
                    break
                
                # Simulate user message
                user_response, new_facts = create_simulated_chat(
                    messages, 
                    is_user=True,
                    facts=person_facts,
                    completed_facts=completed_facts
                )
                
                completed_facts.extend(new_facts)
                completed_facts = list(set(completed_facts))
                
                user_message = {"content": user_response, "isUser": True}
                messages.append(user_message)            
                # Simulate life coach response
                coach_response, _ = create_simulated_chat(
                    messages, 
                    is_user=False,
                )
                
                coach_message = {"content": coach_response, "isUser": False}
                messages.append(coach_message)

        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
            break  # Exit the entire simulation
        except Exception as e:
            print(f"\nError during simulation for person {person_id}: {str(e)}")
            continue  # Move to next person
        finally:
            save_conversation(messages, person_id)
            print(f"\nCompleted simulation for person {person_id}")

if __name__ == "__main__":
    # You can modify this list to include any person IDs you want to simulate
    person_ids = [6,7,8,9,10]
    run_simulation(person_ids)