import json
from typing import Dict, List, Set, TypedDict, Tuple
from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
import inquirer

@dataclass
class QuestionMetrics:
    precision: float
    recall: float
    f1: float

class DifficultyStats(TypedDict):
    correct: int
    total: int
    accuracy: float

class QuizMetrics:
    def __init__(self, person_data: Dict):
        self.person_data = person_data
        self.metrics_by_difficulty = defaultdict(lambda: defaultdict(float))
        self.overall_metrics = defaultdict(float)
        self.question_counts = self._init_question_counts()
        
    def _init_question_counts(self) -> Dict:
        return {
            "overall": 0,
            "by_difficulty": defaultdict(int)
        }
    
    def calculate_all_metrics(self) -> Dict:
        """Calculate all metrics for the quiz."""
        self._process_all_questions()
        self._calculate_averages()
        
        return {
            "overall": dict(self.overall_metrics),
            "by_difficulty": dict(self.metrics_by_difficulty)
        }
    
    def _process_all_questions(self) -> None:
        """Process metrics for all questions in the quiz."""
        for question in self.person_data["questions"]:
            self._process_single_question(question)
    
    def _process_single_question(self, question: Dict) -> None:
        """Calculate and store metrics for a single question."""
        difficulty = question["difficulty"]
        metrics = self._calculate_question_metrics(
            question["predicted_memory_ids"],
            question["actual_memory_ids"]
        )
        
        self._update_metrics(metrics, difficulty)
        self._update_counts(difficulty)
    
    def _calculate_question_metrics(self, predicted_ids: List[int], actual_ids: List[int]) -> QuestionMetrics:
        """Calculate precision, recall, and F1 score for a single question."""
        predicted_set = set(predicted_ids)
        actual_set = set(actual_ids)
        
        true_positives = len(predicted_set.intersection(actual_set))
        
        precision = true_positives / len(predicted_set) if predicted_set else 0
        recall = true_positives / len(actual_set) if actual_set else 0
        f1 = self._calculate_f1(precision, recall)
        
        return QuestionMetrics(precision=precision, recall=recall, f1=f1)
    
    @staticmethod
    def _calculate_f1(precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall."""
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    
    def _update_metrics(self, metrics: QuestionMetrics, difficulty: str) -> None:
        """Update running totals for metrics."""
        for metric_name, value in metrics.__dict__.items():
            self.overall_metrics[metric_name] += value
            self.metrics_by_difficulty[difficulty][metric_name] += value
    
    def _update_counts(self, difficulty: str) -> None:
        """Update question counts for averaging."""
        self.question_counts["overall"] += 1
        self.question_counts["by_difficulty"][difficulty] += 1
    
    def _calculate_averages(self) -> None:
        """Calculate average metrics across all questions."""
        metrics_list = ["precision", "recall", "f1"]
        
        # Calculate overall averages
        for metric in metrics_list:
            self.overall_metrics[metric] /= self.question_counts["overall"]
        
        # Calculate averages by difficulty
        for difficulty in self.metrics_by_difficulty:
            question_count = self.question_counts["by_difficulty"][difficulty]
            for metric in metrics_list:
                self.metrics_by_difficulty[difficulty][metric] /= question_count

class MetricsReporter:
    def __init__(self, all_metrics: Dict, quiz_data: List):
        self.all_metrics = all_metrics
        self.quiz_data = quiz_data
        self.aggregate_metrics = self._calculate_aggregate_metrics()
    
    def _calculate_aggregate_metrics(self) -> Dict:
        """Calculate average metrics across all people"""
        total_metrics = {
            "overall": defaultdict(float),
            "by_difficulty": defaultdict(lambda: defaultdict(float))
        }
        
        # Sum up all metrics
        for metrics in self.all_metrics.values():
            for metric, value in metrics["overall"].items():
                total_metrics["overall"][metric] += value
                
            for difficulty, diff_metrics in metrics["by_difficulty"].items():
                for metric, value in diff_metrics.items():
                    total_metrics["by_difficulty"][difficulty][metric] += value
        
        # Calculate averages
        num_people = len(self.all_metrics)
        for metric in total_metrics["overall"]:
            total_metrics["overall"][metric] /= num_people
            
        for difficulty in total_metrics["by_difficulty"]:
            for metric in total_metrics["by_difficulty"][difficulty]:
                total_metrics["by_difficulty"][difficulty][metric] /= num_people
        
        return {
            "overall": dict(total_metrics["overall"]),
            "by_difficulty": {k: dict(v) for k, v in total_metrics["by_difficulty"].items()}
        }
    
    def print_report(self) -> None:
        """Print a formatted report of all metrics."""
        # Print individual metrics
        for person_id, metrics in self.all_metrics.items():
            print(f"\n=== Metrics for Person {person_id} ===")
            self._print_person_metrics(metrics)
        
        # Print aggregate metrics
        print("\n=== Aggregate Metrics (Average Across All People) ===")
        self._print_person_metrics(self.aggregate_metrics)
    
    def _print_person_metrics(self, metrics: Dict) -> None:
        print("\nOverall:")
        for metric, value in metrics["overall"].items():
            print(f"{metric.capitalize()}: {value:.3f}")
        
        print("\nMetrics by Difficulty:")
        for difficulty, diff_metrics in metrics["by_difficulty"].items():
            print(f"\n{difficulty.upper()}:")
            for metric, value in diff_metrics.items():
                print(f"{metric.capitalize()}: {value:.3f}")
    
    def print_worst_performing_questions(self):
        """Print the worst performing questions by recall and precision"""
        # Load extracted_memories data to get memory contents
        with open("data/extracted_memories.json", 'r') as f:
            extracted_memories = json.load(f)
        
        # Create a mapping of memory id to content for quick lookup
        memory_map = {}
        for person in extracted_memories:
            for memory in person["extracted_memories"]:
                # Handle both single and multiple IDs
                for memory_id in memory["id"]:
                    memory_map[memory_id] = memory["content"]
        
        # Flatten all questions with their metrics
        question_metrics = []
        
        for person in self.quiz_data:
            person_id = person["person_id"]
            for q in person["questions"]:
                predicted_set = set(q["predicted_memory_ids"])
                actual_set = set(q["actual_memory_ids"])
                
                true_positives = len(predicted_set & actual_set)
                precision = true_positives / len(predicted_set) if predicted_set else 0
                recall = true_positives / len(actual_set) if actual_set else 0
                
                question_metrics.append({
                    'person_id': person_id,
                    'question': q["question"],
                    'precision': precision,
                    'recall': recall,
                    'predicted_ids': q["predicted_memory_ids"],
                    'predicted_texts': q["predicted_texts"],
                    'actual_ids': q["actual_memory_ids"]
                })
        
        # Sort by recall and precision
        worst_recall = sorted(question_metrics, key=lambda x: x['recall'])[:10]
        worst_precision = sorted(question_metrics, key=lambda x: x['precision'])[:10]
        
        # Print worst recall questions
        print("\n" + "="*100)
        print("WORST 10 QUESTIONS BY RECALL")
        print("="*100)
        for i, q in enumerate(worst_recall, 1):
            print(f"\n{i}. Question (Person {q['person_id']}): {q['question']}")
            print(f"   Recall Score: {q['recall']:.3f}")
            print(f"   Precision Score: {q['precision']:.3f}")
            print("\n   Predicted Memories (What the model matched):")
            for pid, ptext in zip(q['predicted_ids'], q['predicted_texts']):
                print(f"   - ID {pid}: {ptext}")
            print("\n   Actual Memories (What should have been matched):")
            for aid in q['actual_ids']:
                content = memory_map.get(aid, f"Memory {aid} not found")
                print(f"   - ID {aid}: {content}")
            print("-"*80)
        
        # Print worst precision questions
        print("\n" + "="*100)
        print("WORST 10 QUESTIONS BY PRECISION")
        print("="*100)
        for i, q in enumerate(worst_precision, 1):
            print(f"\n{i}. Question (Person {q['person_id']}): {q['question']}")
            print(f"   Precision Score: {q['precision']:.3f}")
            print(f"   Recall Score: {q['recall']:.3f}")
            print("\n   Predicted Memories (What the model matched):")
            for pid, ptext in zip(q['predicted_ids'], q['predicted_texts']):
                print(f"   - ID {pid}: {ptext}")
            print("\n   Actual Memories (What should have been matched):")
            for aid in q['actual_ids']:
                content = memory_map.get(aid, f"Memory {aid} not found")
                print(f"   - ID {aid}: {content}")
            print("-"*80)

def analyze_quiz(quiz_file_path: str) -> Tuple[Dict, List]:
    """Main function to analyze a completed memory quiz."""
    with open(quiz_file_path, 'r') as f:
        quiz_data = json.load(f)
    
    all_metrics = {}
    for person in quiz_data:
        person_id = person["person_id"]
        quiz_metrics = QuizMetrics(person)
        all_metrics[person_id] = quiz_metrics.calculate_all_metrics()
    
    return all_metrics, quiz_data

def get_test_files() -> List[str]:
    """Get all JSON files in the data/tests directory, sorted by newest first."""
    tests_dir = Path("data/tests")
    if not tests_dir.exists():
        print("Error: data/tests directory not found")
        return []
    
    # Get all JSON files with their modification times
    files_with_times = [(f, f.stat().st_mtime) for f in tests_dir.glob("*.json")]
    # Sort by modification time (newest first) and extract just the file paths
    sorted_files = [str(f) for f, _ in sorted(files_with_times, key=lambda x: x[1], reverse=True)]
    
    return sorted_files

def select_test_file(test_files: List[str]) -> str:
    """Allow user to select a test file using arrow keys."""
    if not test_files:
        print("No test files found in data/tests directory")
        exit(1)
    
    questions = [
        inquirer.List('file',
                     message="Select a file to analyze",
                     choices=[Path(f).name for f in test_files],
                     carousel=True)
    ]
    
    answers = inquirer.prompt(questions)
    if not answers:  # User pressed Ctrl+C
        exit(0)
        
    selected_name = answers['file']
    return next(f for f in test_files if Path(f).name == selected_name)

def main():
    test_files = get_test_files()
    selected_file = select_test_file(test_files)
    
    print(f"\nAnalyzing: {Path(selected_file).name}")
    all_metrics, quiz_data = analyze_quiz(selected_file)
    
    reporter = MetricsReporter(all_metrics, quiz_data)
    reporter.print_report()
    reporter.print_worst_performing_questions()

if __name__ == "__main__":
    main() 