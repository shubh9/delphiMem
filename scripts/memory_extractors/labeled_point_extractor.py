from typing import List, Dict
from openai import AzureOpenAI
import json
import os
from dotenv import load_dotenv

class LabeledPointExtractor:
    MEMORY_EXTRACTION_PROMPT = """
    Analyze the following message and previous context to extract any new personal information or memories about the speaker.
    Do not extract information that is already in the existing memories.

    Format the response as a JSON array of memories, where each memory is an object with:
    - "content": The actual memory/information about the user. 

    Labeling:
    Also label every noun in the memory and reword memory to always include the name and then the label.
    For example, if the message is, my daughter likes the color pink, return "Emma<daughter> likes the color pink"

    Don't talk in first person! Always use both Names and labels.
    If the name isn't known use the exact string [USER_NAME_PLACEHOLDER] instead. 
    
    For example for the message. Playing tennis is fun on the weekends. 
    return "Bob<user> plays tennis on the weekends" or if the name isn't known "[USER_NAME_PLACEHOLDER]<user> plays tennis on the weekends"

    Or for the message, "My daughter feels sick"
    return "Emma<daughter> feels sick" or if the name isn't known "[DAUGHTER_NAME_PLACEHOLDER]<daughter> feels sick"

    Do not label obvious things like 24<age>, red<color>, etc. It's moreso things that if a random person was reading it would be unclear what the noun is.

    Add labels in line surrounded by <>. Only add labels to nouns.
    Just print the memory, be concise and don't include any other text.
    Output in a json array.

    When the relationship is know, labels should be in the perspective of the user. 
    Example:
    [{"content": "Lives in San Francisco<city>"}, {"content": "Name is Bob<user>"}, {"content": "Bob<user> is taller than Gerald<friend>"}]

    Instructions:
    Only include clear, specific information.
    Make sure to also include things about the user like name, age, career, friends, hobbies, likes/dislikes, etc. As well as things that the user tells about others like their names, ages, careers, and their relationships with them etc.

    Dont include information that is abstract or just venting. Only include facts and specific information about the user.
    NEGATIVE EXAMPLES:
    - "the user's name is common" (interesting fact rather than information about the user)
    - is very disiplined at work at dribble<company> (no label and not a specific chatacter atribute about the user)
    - hates team sports which is why he like working in small teams (no label and too long and vague, none of it should be included!)
    - gets drinks with friends every weekend (no label and not a specific chatacter atribute about the user)
    - {"content": "[USER_NAME_PLACEHOLDER]<user> has a demanding job"}
    - Thinks vegetarians are weird (not specific, not a fact about the user!)

    POSITIVE EXAMPLES:
    - "Navid<user> lives in San Francisco<city>"
    - "Bob<user> works out every morning"
    - "Navid<user> is a workaholic"
    - "Sam<user> works at Dribble<company>"
    - "Will<user> is a software engineer<profession>"
    - "Navid<user> went on a date with Sarah<romantic_interest>" last week"

    Functions:
    In the array you can also return functions. There are one functions you can return:
    Setting Variables. Something like USER_NAME_PLACEHOLDER is a variable. Once the value of this is found return a function that sets the variable to the value.
    Example: For the message "people say to me all the time 'Greg can you eat pork' and have to tell them that i can't"
    [
        {"content": "Greg<user> cannot eat pork"},
        {"function": "[USER_NAME_PLACEHOLDER] = Greg"}
    ]

    Example 2: For the message "My friends always tell me how smart Emma is what can you say the apple doesn't fall far from the tree"
    [
        {"content": "Emma<daughter> is smart"},
        {"function": "[DAUGHTER_NAME_PLACEHOLDER] = Emma"}
    ]

    - 
    If no new memories are found, return an empty array.
    """

    def __init__(self):
        load_dotenv()
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-15-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.variable_replacements = {}

    def get_message_context(self, messages: List[Dict], current_index: int) -> str:
        """Get the last 3 messages before the current message for context"""
        start_idx = max(0, current_index - 3)
        context_messages = messages[start_idx:current_index]
        return "\n".join([f"{'User' if msg['isUser'] else 'AI'}: {msg['content']}" for msg in context_messages])

    def apply_variable_replacements(self, memory: Dict) -> Dict:
        """Apply stored variable replacements to memory content"""
        if 'content' in memory:
            for placeholder, value in self.variable_replacements.items():
                memory['content'] = memory['content'].replace(placeholder, value)
        return memory

    def process_function(self, memory: Dict):
        """Process function memories to store variable replacements"""
        if 'function' in memory:
            try:
                # Parse function of format "[PLACEHOLDER] = Value"
                placeholder, value = memory['function'].split('=')
                placeholder = placeholder.strip()
                value = value.strip()
                self.variable_replacements[placeholder] = value
            except Exception as e:
                print(f"Error processing function: {str(e)}")

    def extract_memories(self, messages: List[Dict], person_id: str) -> List[Dict]:
        """Extract memories for a single person using the base point method"""
        person_memories = []
        
        # Only analyze user messages
        user_messages = [msg for msg in messages if msg['isUser']]
        
        for i, message in enumerate(user_messages):
            # Find the index of this message in the full conversation
            message_idx = messages.index(message)
            context = self.get_message_context(messages, message_idx)
            
            # Format existing memories for the prompt
            existing_memories_str = json.dumps(person_memories, indent=2) if person_memories else ""
            
            prompt = self.MEMORY_EXTRACTION_PROMPT + f"""
                {f"\n\nPrevious messages for context:\n{context}" if context else ""}
                {f"\n\nBelow is the infromation we already know about the user, make sure not to repeat any of this information:\n{existing_memories_str}" if existing_memories_str else ""}
                \n\nMessage to analyze:\n{message['content']}
            """

            # print(f"Analyzing message {i+1}/{len(user_messages)} for person {person_id}")            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user", 
                    "content": prompt
                }]
            )
            
            # Clean the response content to handle ```json wrapped responses
            content = response.choices[0].message.content
            if '[' in content and ']' in content:
                start = content.find('[')
                end = content.rfind(']') + 1
                content = content[start:end]

            print(f"\n=== MESSAGE {i+1}/{len(user_messages)} ===")
            print(message['content'])
            print("========================\n")

            print("\n=== EXTRACTED MEMORIES ===")
            print(content)
            print("========================\n")

            try:
                memories = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print("Response content:", content)
                memories = []  # Default to an empty list if parsing fails

            if memories:
                for memory in memories:
                    # Process functions to store variable replacements
                    if 'function' in memory:
                        self.process_function(memory)
                        updated_memories = []
                        for prev_memory in person_memories:
                            updated_memory = self.apply_variable_replacements(prev_memory.copy())
                            updated_memories.append(updated_memory)
                        person_memories = updated_memories
                        continue

                    else:
                        # Apply any stored replacements to the memory content
                        memory = self.apply_variable_replacements(memory)
                        # Only add new, unique memories
                        if memory not in person_memories:
                            person_memories.append(memory)
        
        return person_memories 