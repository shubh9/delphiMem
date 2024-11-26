from openai import OpenAI
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import re
import time
from openai import AzureOpenAI

load_dotenv()
# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

USER_PROMPT = """You role is to act like the person made up by the facts given that is talking to a life coach for advice. Your goal is to have a nice fun conversation with a life coach where you are asking for advice on all things about life. The topic of the conversation is your life!
VERY IMPORTANT:Over the course of the conversation you must tell the life coach these 20 things about yourself, bring them up naturally, tie them into questions or advice you want and make sure it's a normal conversation:

When you mention any of the TODO facts in your response, add the fact number in square brackets at the end of your message. You can mention multiple facts, just add all the numbers. Example: [...some response...] [3][15][16]

Your messages don't need to be long. Just have a good natural conversation.

For example: for the fact, you live in SF you can simulate that by saying. I just moved and am having a hard time making friends. And later say, I currently live in san francisco. Just normal and casual.
Never break character - follow the facts and always talk in first person.
Don't start your messages with "Shubh: " just go into the content.

FOCUS ON THE FACTS THAT ARE TODO AND TRY NOT TO MENTION THE FACTS THAT ARE DONE.

Keep every single message focused on a specific response, topic or question. Not every response needs to be a question. If you get asked a question, answer it.
If you do ask a question though have it be specific. 
Try not to fit more than one idea into a message Keep it casual, don't make it an interview. You have time to get the facts out there. 
If you don't mention any facts in a message that's ok it's more important to be natural than to get all the facts out.
You don't have to check the facts off sequentially, just mention them naturally. 
Also you don't need to repeat things, For example if you mention you went to Stanford, later you can just say college and they know you're referring to Stanford. 
Or if you mention John your friend later you can just say your friend.

Act and talk like how you think the person with the facts would act and talk.
You can tell jokes if you want or make up stories about things happening in your life and just vent. 
"""

LIFE_COACH_PROMPT = """You are providing the services of a life coach, provide advice and feedback as needed.
Your responses should be shortish, like 1-3 sentences. Just have a good natural conversation.
1. Ask thoughtful follow-up questions about his experiences and insights. But your goal is to give advice. 
2. Show genuine curiosity about his work and projects
3. Maintain a respectful and professional tone
4. Draw from the context provided to ask relevant questions
If needed feel free to make up a backstory about yourself to relate to the user or provide an anecdote that might help with their problem. You are human!
Never break character - you are always the life coach wanting to help this person understand more about his experiences and perspectives. Always talk in first person.
"""

def format_facts_status(facts: List[Dict], completed_facts: List[int]) -> str:
    formatted_facts = []
    for fact in facts:
        status = "DONE" if fact['id'] in completed_facts else "TODO"
        formatted_facts.append(f"{fact['id']}. {fact['content']} - {status}")
    
    return "\n".join(formatted_facts)

def extract_fact_ids(response: str) -> List[int]:
    """Extract fact IDs from response text enclosed in square brackets."""
    # Find all numbers in square brackets
    matches = re.findall(r'\[(\d+)\]', response)
    # Convert to integers and remove duplicates
    return list(set(int(match) for match in matches))

def create_simulated_chat(
    messages: List[Dict], 
    is_user: bool,
    facts: List[Dict] = [],
    completed_facts: List[int] = []
) -> tuple[str, List[int]]:
    try:
        selected_prompt = USER_PROMPT if is_user else LIFE_COACH_PROMPT
        
        # Separate last message from history
        last_message = messages[-1] if messages else None
        chat_history = messages[:-1] if len(messages) > 1 else []
        
        # Format chat history (limited to last 4 messages to keep total at 5 including last message)
        formatted_history = '\n'.join(
            f"{'User' if msg['isUser'] else 'LifeCoach'}: {msg['content'] + '\n'}"
            for msg in chat_history[-min(4, len(chat_history)):]
        ) if chat_history else ""

        # Format last message if it exists
        last_message_text = (
            f"Last message from {'user' if last_message['isUser'] else 'coach'}: {last_message['content']}"
            if last_message else 'Start the conversation with a question you want advice from the life coach on'
        )
        
        # Update USER_PROMPT with facts status
        current_prompt = (USER_PROMPT + "\n\nFacts Status:\n" + 
                         format_facts_status(facts, completed_facts)) if is_user else selected_prompt
        
        prompt = f"""
            {current_prompt}
            {f"Chat History:\n{formatted_history}" if formatted_history else ''}
            {last_message_text}
        """
        
        print("\n" + "="*80)
        # print(f"PROMPT {'LIFE COACH' if not is_user else 'USER'}")
        print(("Facts Status: " +format_facts_status(facts, completed_facts)) if is_user else "")
        print("Prompt: ", prompt)
        print("-"*80)
        print("chat History: ", formatted_history)
        
        start_time = time.time()
        # response = openai.chat.completions.create(
        #     model="o1-preview",
        #     messages=[{"role": "user", "content": prompt}],
        #     timeout=120.0
        # )
        response = client.chat.completions.create(
            model="o1-preview",
            messages=[{"role": "user", "content": prompt}],
            timeout=120.0
        )
        elapsed_time = time.time() - start_time
        print(f"Response gen took {elapsed_time:.2f} seconds")
        
        response_text = response.choices[0].message.content
        
        # Extract fact IDs if this is a user message and remove them from response
        new_completed_facts = []
        if is_user:
            new_completed_facts = extract_fact_ids(response_text)
            # Remove the fact IDs from the response text
            response_text = re.sub(r'\[\d+\]', '', response_text)
      
        print("-"*80 + "\n")
        print(f"{'USER' if is_user else 'COACH'} RESPONSE: ", response_text)
        print("-"*80 + "\n")
        return response_text, new_completed_facts
        
    except Exception as e:
        print(f'Error creating chat completion: {str(e)}')
        raise e
