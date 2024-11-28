import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from openai import OpenAI, AzureOpenAI
from dataclasses import dataclass
from tqdm import tqdm
from dotenv import load_dotenv
import os
import argparse
import datetime
import glob
import inquirer

# Load environment variables
load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

@dataclass
class MemoryQuizQuestion:
    id: int
    question: str
    right_memory_ids: List[int]
    difficulty: str

@dataclass
class Memory:
    id: int
    text: str
    category: str  # Added to track which profile category the memory belongs to
    embedding: List[float]

@dataclass
class QuizResult:
    question_id: int
    question: str
    predicted_memory_ids: List[int]
    actual_memory_ids: List[int]
    predicted_texts: List[str]
    difficulty: str

@dataclass
class EntityInfo:
    entity_id: int
    descriptions: str

@dataclass
class LLMQueryResponse:
    entity_id: int
    attribute: str

class StructuredMemoryQuizEvaluator:
    def __init__(self, memories_file: str = None):
        self.memories_file = memories_file or self.pick_memories_file()
        self.available_person_ids = self._get_available_person_ids()
        self.questions_by_person = self._load_questions()
        self.attribute_embeddings: Dict[str, Dict[int, List[Dict]]] = {}

    def _get_available_person_ids(self) -> List[int]:
        """Get list of entity IDs from the memories file"""
        with open(self.memories_file, 'r') as f:
            data = json.load(f)
        
        # Get all entity IDs from the extracted_memories array
        person_ids = []
        for person in data:
            person_ids.append(person["person_id"])
        return person_ids

    def _load_questions(self) -> Dict[int, List[MemoryQuizQuestion]]:
        """Load questions from memory_quiz.json only for available person IDs"""
        quiz_path = Path("data/memory_quiz.json")
        with open(quiz_path, 'r') as f:
            data = json.load(f)
        
        questions_by_person = {}
        for person_data in data:
            person_id = person_data["person_id"]
            # Only load questions for people who exist in the memories file
            if person_id in self.available_person_ids:
                questions = [
                    MemoryQuizQuestion(
                        id=q["id"],
                        question=q["question"],
                        right_memory_ids=q["right_memory_ids"],
                        difficulty=q["difficulty"]
                    )
                    for q in person_data["questions"]
                ]
                questions_by_person[person_id] = questions
            
        return questions_by_person

    def pick_memories_file(self) -> str:
        """Display an inquirer picker interface for selecting the memories file"""
        files = glob.glob("data/extracted_memories/*.json")
        
        if not files:
            raise ValueError("No JSON files found in data/extracted_memories/")
        
        # Sort files by modification time, most recent first
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Format choices to show both filename and last modified date
        choices = []
        for f in files:
            mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(f))
            mod_time_str = mod_time.strftime("%Y-%m-%d %H:%M")
            basename = os.path.basename(f)
            display_name = f"{basename} (modified: {mod_time_str})"
            choices.append((display_name, f))
        
        questions = [
            inquirer.List('file',
                         message="Select the extracted memories file to use",
                         choices=choices,
                         carousel=True)
        ]
        
        answers = inquirer.prompt(questions)
        
        if not answers:
            raise KeyboardInterrupt("User cancelled file selection")
            
        return answers['file']

    def _load_memories(self, person_id: int) -> None:
        """Load and compute embeddings for all attributes of each entity under a person"""
        with open(self.memories_file, 'r') as f:
            data = json.load(f)

        print(f"Loading memories for person_id {person_id}")
        
        # Find the person's data
        person_data = None
        for person in data:
            if person["person_id"] == person_id:
                person_data = person
                break

        if not person_data:
            raise ValueError(f"No data found for person_id {person_id}")

        # Initialize the attribute embeddings structure
        self.attribute_embeddings = {}
        
        # Process each entity under this person with a progress bar
        for entity in tqdm(person_data["extracted_memories"], desc="Processing Entities"):
            entity_id = entity["Id"]
            self.attribute_embeddings[entity_id] = {}
            
            # Process each attribute in the entity's Profile
            profile = entity.get("Profile", {})
            for attribute, memory_list in profile.items():
                self.attribute_embeddings[entity_id][attribute] = []
                for memory in memory_list:
                    if "content" in memory:
                        embedding = self.generate_embedding(memory["content"])
                        self.attribute_embeddings[entity_id][attribute].append({
                            "text": memory["content"],
                            "embedding": embedding,
                            "ids": memory.get("mem_id", [])
                        })

        # print(f"Attribute Embeddings: {self.attribute_embeddings}")

    def clean_and_validate_json_response(self, response_content: str) -> Dict[str, Any]:
        """Validate and clean up the JSON response from the LLM."""
        try:
            # Attempt to find the JSON object within the response content
            start_index = response_content.find('{')
            end_index = response_content.rfind('}') + 1
            if start_index == -1 or end_index == -1:
                raise ValueError("No valid JSON object found in the response.")
            
            # Extract and parse the JSON object
            json_str = response_content[start_index:end_index]
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e} \n\nResponse content: {json_str}")


    def query_llm_for_attribute(self, question: str, entities: List[EntityInfo]) -> LLMQueryResponse:
        """Query LLM to determine which entity and attribute to search"""

        # Construct the prompt
        entities_desc = "\n\n".join([
            f"Entity {e.entity_id}:\n" + f"Description: {e.descriptions}"
            for e in entities
        ])

        # print("entities: ", entities_desc)

        attributes = [
            "name", "age", "job", "location", "health", "interests", "notes"
        ]
        
        prompt = f"""You will be given a question and a list of entities and their descriptions. 
        Your job is to determine which entity and which attribute would contain the answer.
        Return your response in this exact format:
        {{
            "entity_id": <id>,
            "attribute": "<attribute_name>"
        }}

        Given the following entities and their descriptions:
        {entities_desc}
        And these possible attributes:
        {", ".join(attributes)}

        For this question: "{question}"
        """

    

        # print(f"\nPrompt being sent to LLM:")
        # print("-" * 80)
        # print(prompt)
        # print("-" * 80)
        # print()

        response = azure_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes questions and determines which entity and attribute would contain the answer."},
                {"role": "user", "content": prompt}
            ],
        )

        print(f"LLM Response: {response.choices[0].message.content}")

        result = self.clean_and_validate_json_response(response.choices[0].message.content)
        # print(f"LLM Query Response: {result}")
        return LLMQueryResponse(
            entity_id=result["entity_id"],
            attribute=result["attribute"]
    )

    def find_best_memory_match(self, question: str, entity_id: int, attribute: str) -> Tuple[Optional[str], Optional[List[int]]]:
        """Find the best matching memory from a specific attribute of an entity"""
        if entity_id not in self.attribute_embeddings:
            print(f"Entity {entity_id} not found in embeddings")
            return None, None
        if attribute not in self.attribute_embeddings[entity_id]:
            print(f"Attribute '{attribute}' not found for entity {entity_id}")
            return None, None
        print(f"Finding best memory match for question: {question}, entity_id: {entity_id}, attribute: {attribute}")
        question_embedding = self.generate_embedding(question)
        memories = self.attribute_embeddings[entity_id][attribute]
        
        best_similarity = -1
        best_memory = None
        best_memory_ids = None
        
        for memory in memories:
            similarity = self.compute_similarity(question_embedding, memory["embedding"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_memory = memory["text"]
                best_memory_ids = memory.get("ids", [])
        
        print(f"Best memory match: {best_memory} with ids: {best_memory_ids} and similarity: {best_similarity}")
        return best_memory, best_memory_ids

    def evaluate_person(self, person_id: int) -> List[QuizResult]:
        """Evaluate all questions for a given person"""
        if person_id not in self.questions_by_person:
            raise ValueError(f"No questions found for person_id {person_id}")
        
        # Load memories for this person
        self._load_memories(person_id)
        
        # Get all entities for this person
        with open(self.memories_file, 'r') as f:
            data = json.load(f)
        
        # Find all entities for this person and their descriptions
        entities = []
        for person in data:
            if person["person_id"] == person_id:
                for entity in person["extracted_memories"]:
                     # Add profile information to descriptions
                    if entity["Id"] in self.attribute_embeddings:
                        entity_info = EntityInfo(
                            entity_id=entity["Id"],
                            descriptions=entity['Description']
                        )
                        entities.append(entity_info)
                break
        
        results = []
        questions = self.questions_by_person[person_id]
        
        for question in tqdm(questions, desc=f"Evaluating Person {person_id}"):
            search_query = self.query_llm_for_attribute(
                question.question,
                entities
            )
            
            best_match, predicted_ids = self.find_best_memory_match(
                question.question,
                search_query.entity_id,
                search_query.attribute
            )
            
            result = QuizResult(
                question_id=question.id,
                question=question.question,
                predicted_memory_ids=[predicted_ids] if predicted_ids else [],
                actual_memory_ids=question.right_memory_ids,
                predicted_texts=[best_match] if best_match else [],
                difficulty=question.difficulty
            )
            results.append(result)
            
        return results

    def get_entity_descriptions(self, entity_id: int) -> List[str]:
        """Get descriptions for an entity from their memories"""
        with open(self.memories_file, 'r') as f:
            data = json.load(f)
        
        descriptions = []
        
        # Find the entity and get its main Description
        for person in data:
            for entity in person["extracted_memories"]:
                if entity["Id"] == entity_id:
                    descriptions.append(f"Description: {entity['Description']}")
                    break

        # Add profile information
        if entity_id in self.attribute_embeddings:
            for attribute, memories in self.attribute_embeddings[entity_id].items():
                for memory in memories:
                    descriptions.append(f"{attribute}: {memory['text']}")
        return descriptions

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a given text using OpenAI's embedding model"""
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
        # return [1]

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        a = np.array(embedding1)
        b = np.array(embedding2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def save_results(self, all_results: Dict[int, List[QuizResult]], output_path: str):
        """Save results to JSON file"""
        formatted_results = []
        
        for person_id, results in all_results.items():
            formatted_results.append({
                "person_id": person_id,
                "questions": [
                    {
                        "question_id": r.question_id,
                        "question": r.question,
                        "difficulty": r.difficulty,
                        "predicted_memory_ids": r.predicted_memory_ids,
                        "actual_memory_ids": r.actual_memory_ids,
                        "predicted_texts": r.predicted_texts
                    }
                    for r in results
                ]
            })
            
        with open(output_path, 'w') as f:
            json.dump(formatted_results, f, indent=2)

def main():
    evaluator = StructuredMemoryQuizEvaluator()
    
    print(f"\nProcessing memories file: {os.path.basename(evaluator.memories_file)}")
    print(f"Found {len(evaluator.available_person_ids)} people in memories file")
    print(f"Person IDs: {evaluator.available_person_ids}")
    
    all_results = {}
    for person_id in evaluator.available_person_ids:
        if person_id in evaluator.questions_by_person:
            print(f"\nProcessing person {person_id}...")
            results = evaluator.evaluate_person(person_id)
            all_results[person_id] = results
        else:
            print(f"\nSkipping person {person_id} - no questions found in quiz file")
    
    memories_basename = os.path.splitext(os.path.basename(evaluator.memories_file))[0]
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    output_path = f"data/tests/structured_memory_quiz_{memories_basename}_{current_date}.json"
    evaluator.save_results(all_results, output_path)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main() 