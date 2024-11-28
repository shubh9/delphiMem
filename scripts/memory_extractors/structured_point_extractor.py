from typing import List, Dict
from openai import AzureOpenAI
import json
import os
import random
from dotenv import load_dotenv
from .base_point_extractor import BasePointExtractor

class StructuredPointExtractor(BasePointExtractor):
    MEMORY_EXTRACTION_PROMPT = """
    Analyze the following message and previous context to extract structured information about entities (people, places, things) mentioned.
    Format the response as a JSON array where each entry represents an operation on an entity with the following structure:

    - "Entity": The ID of the person (use the person_id provided, if no relevant entity exists create a new ID for the entity and then use that for subsequent messages)
    - "Function": The operation type only 2 types. 
        - CREATE for new entities. When you do this pick a random name like "friend_1" or "family_1" etc. but make sure you're consistent with it for the rest of the functions
        Use CREATE rarely, remember an entity is a person, one singular person. Create an entity for the user's sister, friend, etc but only when you have specific information.
         Like on the message my coworkers are all good too shoes, don't create one because that's not specific, but if they mention a coworker by name then you create one.
         Don't create an entity for "the user's family" or "the user's friends" or "the user's coworkers" etc. Only create an entity for a specific person.
         Also until you have their name, always refer to the person talking to you as "the user"
        - ADD for adding information to existing entities either an attribute or relationship

    Followed by either an Attribute or Relationship:
        - "Attribute": if the type of information being added is about the entity itself Refer to the list of attributes below, only use one of those.
        - "Relationship": if the type of information being added is about the relationship between the entity and another entity (friend, family, coworker, etc.)
        When you add a relationship, make sure you do it twice in both directions. 
        Once a connection is made, don't create it again! Add to an attribute. Relationships should be rarely created, usually just when the entity is created. 
        The content should be in first person,how the other person is related to the entity.
        For example if 25432 is the user and 80047 is the user's son then {"Entity": "25432", "Function": "ADD", "Relationship": "80047", "Content": "this is my son"}

    - "Content": The actual information/value. 

    If it's an attribute then content should be the value of the attribute in whole sentences. 
    If it's a relationship then content should be a short description of nature of the relationship.
    If the function is CREATE then the content should be a short description of the entity.

    If the attribute is "dislike" then the content should be the type of food they like for example. Answer in entire sentences like "finds the city of san francisco annoying" or "pineapple on a perfectly good cheese pizza". 
        If the relationship is "mom" then the content should be self-explanatory like "Mom of Navid". and then add the name attribute to the entity.

        3 types of response: Follow this very strictly!!
    {"Entity": "entity_id", "Function": "ADD", "Relationship": "relationship_id", "Content": "relationship description"}
    {"Entity": "entity_id", "Function": "ADD", "Attribute": "attribute_name", "Content": "attribute value"}
    {"Entity": "entity_id", "Function": "CREATE", "Content": "entity description"}


    Example for the message: "It's crazy that at 25 i'm still able to make friends. I just met John the coolest person at a basketball pickup game)
    Example output: (you will be given the information that the user is the entity with ID "23325 and that the age is not known)
    [
        {"Entity": "23325", "Function": "ADD", "Attribute": "age", "Content": "25 years old"},
        {"Entity": "friend_1", "Function": "CREATE", "Content": "Named John, this is a new friend of the user that plays basketball"},
        {"Entity": "friend_1", "Function": "ADD", "Attribute": "name", "Content": "John Smith"},
        {"Entity": "23325", "Function": "ADD", "Relationship": "friend_1", "Content": "friend from basketball pickup game"},
        {"Entity": "friend_1", "Function": "ADD", "Relationship": "friend_1", "Content": "friend from basketball pickup game"}
    ]

    List of Attributes, ONLY USE ONE OF THESE:
    - name
    - age
    - job
    - location
    - health
    - interests
    - notes

    Rules:
    1. Use CREATE function when introducing a new entity
    2. Use ADD function when adding information to existing entities
    3. Break down complex information into separate entries
    4. Be specific and concise in the Content field
    6. Don't repeat information that's already been captured

    If no new information is found, return an empty array.
    """

    def __init__(self):
        super().__init__()
        self.entities = []  # Store all entities and their information
        
    def generate_entity_id(self) -> str:
        """Generate a new 5-digit entity ID"""
        while True:
            new_id = random.randint(10000, 99999)
            if not any(e['Id'] == new_id for e in self.entities):
                return new_id

    def get_entity_information(self) -> str:
        """Step 1: Extract all known entity information from previous messages"""
        entity_info = []
        for entity in self.entities:
            print(f"Entity {entity['Id']}:, Description: {entity['Description']}")
            info = f"Entity {entity['Id']}:, Description: {entity['Description']}\n"
            if entity['Profile']:
                info += "Profile:\n" + "\n".join([f"- {attr}: {values}" for attr, values in entity['Profile'].items()])
            if entity['Connections']:
                info += "\nConnections:\n" + "\n".join([f"- Connected to {conn['id']}: {conn['relationship']}" for conn in entity['Connections']])
            entity_info.append(info)
        return "\n\n".join(entity_info)

    def process_memory_operation(self, operation: Dict) -> None:
        """Step 2: Process individual memory operations and update the entities store"""
        entity_id = operation['Entity']
        
        if operation['Function'] == 'CREATE':
            # Generate new ID and create entity
            new_id = self.generate_entity_id()
            new_entity = {
                'Id': new_id,
                'Description': operation['Content'],
                'Profile': {},
                'Connections': []
            }
            self.entities.append(new_entity)
            return new_id
            
        elif operation['Function'] == 'ADD':
            # Find entity in the list
            entity = next((e for e in self.entities if str(e['Id']) == str(entity_id)), None)
            if not entity:
                print(f"Error: Entity {entity_id} not found")
                return
                
            if 'Attribute' in operation:
                # Handle attribute addition
                attr_name = operation['Attribute']
                if attr_name not in entity['Profile']:
                    entity['Profile'][attr_name] = []
                
                # Create new attribute entry
                new_attribute = {
                    "content": operation['Content'],
                    "mem_id": []
                }
                
                # Check for duplicates
                existing_contents = [attr['content'] for attr in entity['Profile'][attr_name]]
                if operation['Content'] not in existing_contents:
                    entity['Profile'][attr_name].append(new_attribute)
                    
            elif 'Relationship' in operation:
                # Handle relationship addition
                new_connection = {
                    'id': operation['Relationship'],
                    'relationship': operation['Content']
                }
                # Check for duplicate relationship
                for connection in entity['Connections']:
                    if connection['id'] == new_connection['id'] and connection['relationship'] == new_connection['relationship']:
                        print(f"ERROR: trying to create duplicate relationship for entity {entity_id}: {new_connection}")
                        return
                
                entity['Connections'].append(new_connection)

    def extract_memories(self, messages: List[Dict], person_id: str) -> Dict:
        """Main extraction process"""
        user_messages = [msg for msg in messages if msg['isUser']]
        
        # Initialize entities as a list instead of dict
        self.entities = []
        
        # Create initial user entity
        new_id = self.generate_entity_id()
        initial_user = {
            'Id': new_id,
            'Description': "This is the user, the person that is talking to you",
            'Profile': {},
            'Connections': []
        }
        self.entities.append(initial_user)
        
        for i, message in enumerate(user_messages):
            print(f"Analyzing message {i+1}/{len(user_messages)} for person {person_id}")
            # Step 1: Get current entity information
            entity_info = self.get_entity_information()

            # Get message context
            message_idx = messages.index(message)
            context = self.get_message_context(messages, message_idx)
            
            # Prepare and send prompt - Move the conditional text here
            full_prompt = (
                self.MEMORY_EXTRACTION_PROMPT + 
                (f"\n\nRecent Chat History:\n{context}" if context else "") +
                (f"\n\nRelevant Entity Information:\n{entity_info}" if entity_info else "\n\nNo relevant entities found, should probably create a new entity") +
                f"\n\nMessage to analyze:\n{message['content']}"
            )

            # print("\nPrompt being sent to LLM:")
            # print("=" * 80)
            # print(f"Chat History:\n{context}")
            # print("-" * 80)
            # print(f"\n\nRelevant Entity Information:\n{entity_info}")
            # print("-" * 80)
            # print(f"\n\nMessage to analyze:\n{message['content']}")
            # print("=" * 80)
            response = self.client.chat.completions.create(
                model="o1-preview",
                messages=[{
                    "role": "user", 

                    "content": full_prompt
                }]
            )
            
            try:
                print("\nResponse from LLM:")
                print("-" * 80)
                print(response.choices[0].message.content)
                print("-" * 80)
                print()
                # Clean and validate response
                operations = self.validate_and_clean_response(response.choices[0].message.content)
                
                # Step 2: Process each operation
                id_mapping = {} 

                # Separate CREATE operations from others
                create_operations = [op for op in operations if op['Function'] == 'CREATE']
                other_operations = [op for op in operations if op['Function'] != 'CREATE']

                # Process CREATE operations first
                for operation in create_operations:
                    new_id = self.process_memory_operation(operation)
                    id_mapping[operation['Entity']] = new_id

                # Process other operations
                for operation in other_operations:
                    # Update entity ID if it was just created
                    if operation['Entity'] in id_mapping:
                        operation['Entity'] = id_mapping[operation['Entity']]
                    if 'Relationship' in operation and operation['Relationship'] in id_mapping:
                        operation['Relationship'] = id_mapping[operation['Relationship']]
                    self.process_memory_operation(operation)
                    
            except ValueError as e:
                print(f"Error processing message {i+1}: {e}")
                continue
        
        return self.entities

    def validate_and_clean_response(self, raw_response: str) -> List[Dict]:
        """
        Validates and cleans the response from the LLM.
        Ensures proper format and structure of memory operations.
        """
        # Clean up the response - remove markdown formatting if present
        content = raw_response.strip()
        if '```' in content:
            # Extract content between ```json and ``` if present
            start = content.find('```json\n')
            if start == -1:
                start = content.find('```\n')
            if start != -1:
                start = content.find('\n', start) + 1
                end = content.rfind('```')
                content = content[start:end].strip()
        
        # Extract JSON array if there's extra text
        if '[' in content and ']' in content:
            start = content.find('[')
            end = content.rfind(']') + 1
            content = content[start:end]

        try:
            operations = json.loads(content)
            if not isinstance(operations, list):
                raise ValueError("Response must be a JSON array")

            # Validate each operation
            valid_operations = []
            for op in operations:
                if not isinstance(op, dict):
                    print(f"Error: Skipping invalid operation format: {op}")
                    continue

                # Check required fields
                if 'Entity' not in op or 'Function' not in op:
                    print(f"Error: Missing required fields in operation: {op}")
                    continue

                # Validate Function type
                if op['Function'] not in ['CREATE', 'ADD']:
                    print(f"Error: Invalid Function type in operation: {op}")
                    continue

                # Validate CREATE operation format
                if op['Function'] == 'CREATE':
                    if 'Content' not in op:
                        print(f"Error: CREATE operation missing Content: {op}")
                        continue
                    if 'Attribute' in op or 'Relationship' in op:
                        print(f"Error: CREATE operation should not have Attribute or Relationship: {op}")
                        continue

                # Validate ADD operation format
                if op['Function'] == 'ADD':
                    if 'Attribute' not in op and 'Relationship' not in op:
                        print(f"Error: ADD operation missing Attribute or Relationship: {op}")
                        continue
                    if 'Content' not in op:
                        print(f"Error: ADD operation missing Content: {op}")
                        continue
                    if 'Attribute' in op and 'Relationship' in op:
                        print(f"Error: ADD operation should have either Attribute or Relationship, not both: {op}")
                        continue

                valid_operations.append(op)

            return valid_operations

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in response: {e}")
        except Exception as e:
            raise ValueError(f"Error processing response: {e}")