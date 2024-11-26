from typing import List, Dict
from openai import AzureOpenAI
import json
import os
from dotenv import load_dotenv

class BasePointExtractor:
    MEMORY_EXTRACTION_PROMPT = """
    Analyze the following message and previous context to extract any new personal information or memories about the speaker.
    Do not extract information that is already in the existing memories.

    Format the response as a JSON array of memories, where each memory is an object with:
    - "content": The actual memory/information about the user. 

    Just print the memory, be concise and don't include any other text.
    Output in a json array.

    Example:
    [{"content": "Lives in San Francisco"}, {"content": "Name is Bob"}, {"content": "Is a man"}]

    Only include clear, specific information.
    Make sure to also include things about the user like name, age, career, friends, hobbies, likes/dislikes, etc. As well as things that the user tells about others like their names, ages, careers, and their relationships with them etc.

    Dont include information that is abstract or just venting. 
    Examples of things not to include:
    - "the user's name is common" (interesting fact rather than information about the user)
    - is very disiplined at work at dribble (not a specific chatacter atribute about the user)
    - hates team sports which is why he like working in small teams (too long and vague, none of it should be included)
    - usually works out before work in the morning (instead should be "works out often")
    - gets drinks with friends every weekend (not a specific chatacter atribute about the user)
    - Lives in a busy city like Delhi which can be overwhelming at times (multiple topics should be split into, "Lives in Delhi" and "Finds Delhi to be overwhelming")
    - Friend is a big fan of hockey (not enough information, say "John is a big fan of hockey", if you don't know the friend's name don't include it)

    If no new memories are found, return an empty array.
    """

    def __init__(self):
        load_dotenv()
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-15-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

    def get_message_context(self, messages: List[Dict], current_index: int) -> str:
        """Get the last 3 messages before the current message for context"""
        start_idx = max(0, current_index - 3)
        context_messages = messages[start_idx:current_index]
        return "\n".join([f"{'User' if msg['isUser'] else 'AI'}: {msg['content']}" for msg in context_messages])

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

            print(f"Analyzing message {i+1}/{len(user_messages)} for person {person_id}")
            
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

            memories = json.loads(content)
            if memories:
                # Only add new, unique memories
                for memory in memories:
                    if memory not in person_memories:
                        person_memories.append(memory)
        
        return person_memories 