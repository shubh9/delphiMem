import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from dataclasses import dataclass
from tqdm import tqdm
from dotenv import load_dotenv
import os
import argparse
import datetime
# Load environment variables
load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class MemoryQuizQuestion:
    id: int
    question: str
    right_memory_ids: List[int]
    difficulty: str

@dataclass
class Memory:
    id: List[int]
    text: str
    embedding: List[float]

@dataclass
class QuizResult:
    question_id: int
    question: str
    predicted_memory_ids: List[int]
    actual_memory_ids: List[int]
    predicted_texts: List[str]
    difficulty: str

class MemoryQuizEvaluator:
    def __init__(self):
        self.questions_by_person = self._load_questions()
        self.memories = {} 
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI's API"""
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f'Error generating embedding for text "{text}": {str(e)}')
            raise e

    def _load_questions(self) -> Dict[int, List[MemoryQuizQuestion]]:
        """Load questions from memory_quiz.json"""
        quiz_path = Path("data/memory_quiz.json")
        with open(quiz_path, 'r') as f:
            data = json.load(f)
        
        questions_by_person = {}
        for person_data in data:
            person_id = person_data["person_id"]
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

    def _load_memories(self, person_id: int) -> Dict[str, Memory]:
        """Load memories from extracted_memories.json and generate embeddings for a specific person"""
        memories_path = Path("data/extracted_memories_labeledpointextractor_20241125_2240.json")
        with open(memories_path, 'r') as f:
            data = json.load(f)
        
        # Filter memories for the specific person and flatten the memory list
        person_memories = []
        for entry in data:
            if entry.get("person_id") == person_id:
                person_memories.extend(entry["extracted_memories"])
        
        memories = {}
        for memory in tqdm(person_memories, desc=f"Generating memory embeddings for person {person_id}"):
            embedding = self.generate_embedding(memory["content"])
            # Use content as key since it's unique for each actual memory
            memories[memory["content"]] = Memory(
                id=memory["id"],
                text=memory["content"],
                embedding=embedding
            )
        return memories

    def compute_similarity(self, query_embedding: List[float], memory_embedding: List[float]) -> float:
        """Compute cosine similarity between query and memory embeddings"""
        query_np = np.array(query_embedding)
        memory_np = np.array(memory_embedding)
        return float(np.dot(query_np, memory_np) / 
                    (np.linalg.norm(query_np) * np.linalg.norm(memory_np)))

    def get_top_memories(self, question: str, k: int = 5, min_similarity: float = 0.2, fallback_k: int = 3) -> Tuple[List[int], List[str]]:
        """
        Get up to k similar memories for a question, only including those above min_similarity threshold.
        Falls back to top fallback_k matches if no matches above threshold are found.
        """
        question_embedding = self.generate_embedding(question)
        
        all_similarities = []
        filtered_similarities = []
        for memory in self.memories.values():
            similarity = self.compute_similarity(question_embedding, memory.embedding)
            all_similarities.append((memory.id, similarity, memory.text))
            if similarity >= min_similarity:
                filtered_similarities.append((memory.id, similarity, memory.text))
        
        # Sort both lists by similarity score in descending order
        filtered_similarities.sort(key=lambda x: x[1], reverse=True)
        all_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k matches above threshold, or fallback to top 3 if none found
        top_k = filtered_similarities[:k] if filtered_similarities else all_similarities[:fallback_k]
        
        # Print similarities in a nicely formatted way
        print("\nQuestion:", question)
        if filtered_similarities:
            print(f"\nTop matches (similarity >= {min_similarity}):")
        else:
            print(f"\nNo matches above {min_similarity} threshold. Showing top {fallback_k} matches:")
        print("-" * 80)
        
        memory_ids = []
        memory_texts = []
        
        for i, (mem_ids, similarity, text) in enumerate(top_k, 1):
            print(f"{i}. Memory IDs: {mem_ids}")
            print(f"   Similarity Score: {similarity:.4f}")
            print(f"   Text: {text}")
            print("-" * 80)
            
            # Add all IDs associated with this memory
            memory_ids.extend(mem_ids)
            memory_texts.append(text)
        
        return memory_ids, memory_texts

    def evaluate_person(self, person_id: int) -> List[QuizResult]:
        """Evaluate all questions for a given person"""
        if person_id not in self.questions_by_person:
            raise ValueError(f"No questions found for person_id {person_id}")
        
        # Load memories for this specific person
        self.memories = self._load_memories(person_id)
        
        results = []
        questions = self.questions_by_person[person_id]
        
        for question in tqdm(questions, desc=f"Evaluating Person {person_id}"):
            predicted_memories, predicted_texts = self.get_top_memories(question.question)
            
            result = QuizResult(
                question_id=question.id,
                question=question.question,
                predicted_memory_ids=predicted_memories,
                actual_memory_ids=question.right_memory_ids,
                predicted_texts=predicted_texts,
                difficulty=question.difficulty
            )
            results.append(result)
            
        return results

    def calculate_metrics(self, results: List[QuizResult]) -> Dict:
        """Calculate accuracy metrics by difficulty"""
        metrics = {
            "overall": {"correct": 0, "total": 0},
            "by_difficulty": {
                "easy": {"correct": 0, "total": 0},
                "medium": {"correct": 0, "total": 0},
                "hard": {"correct": 0, "total": 0}
            }
        }
        
        for result in results:
            correct_predictions = len(set(result.predicted_memory_ids) & set(result.actual_memory_ids))
            total_required = len(result.actual_memory_ids)
            
            metrics["overall"]["correct"] += correct_predictions
            metrics["overall"]["total"] += total_required
            
            metrics["by_difficulty"][result.difficulty]["correct"] += correct_predictions
            metrics["by_difficulty"][result.difficulty]["total"] += total_required
        
        # Calculate percentages
        metrics["overall"]["accuracy"] = metrics["overall"]["correct"] / metrics["overall"]["total"]
        for difficulty in metrics["by_difficulty"]:
            if metrics["by_difficulty"][difficulty]["total"] > 0:
                metrics["by_difficulty"][difficulty]["accuracy"] = (
                    metrics["by_difficulty"][difficulty]["correct"] / 
                    metrics["by_difficulty"][difficulty]["total"]
                )
            else:
                metrics["by_difficulty"][difficulty]["accuracy"] = 0
                
        return metrics

    def save_results(self, all_results: Dict[int, List[QuizResult]], output_path: str):
        """Save results to JSON file as an array of elements with person_id and questions"""
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
                        "predicted_texts": r.predicted_texts,
                        "accuracy": len(set(r.predicted_memory_ids) & set(r.actual_memory_ids)) / len(r.actual_memory_ids)
                    }
                    for r in results
                ]
            })
            
        with open(output_path, 'w') as f:
            json.dump(formatted_results, f, indent=2)

def main():
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Evaluate memory quiz for a specific person')
    parser.add_argument('--person_id', type=int, help='ID of the person to evaluate')
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = MemoryQuizEvaluator()

    person_ids = list(evaluator.questions_by_person.keys())
    all_results = {}
    for person_id in person_ids:
        results = evaluator.evaluate_person(person_id)
        all_results[person_id] = results
    # Save results
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    output_path = f"data/tests/completed_memory_quiz_{current_date}.json"
    evaluator.save_results(all_results, output_path)
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
