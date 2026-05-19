"""
Evaluation metrics and testing utilities.
"""

from typing import List, Dict
import json


class QAEvaluator:
    """Evaluate QA system performance."""
    
    @staticmethod
    def evaluate_response(response: str, expected_keywords: List[str]) -> Dict[str, float]:
        """
        Simple keyword-based evaluation.
        
        Args:
            response: Generated response
            expected_keywords: List of keywords that should appear
            
        Returns:
            Dictionary with evaluation metrics
        """
        response_lower = response.lower()
        
        found_keywords = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
        coverage = found_keywords / len(expected_keywords) if expected_keywords else 0
        
        return {
            "coverage": coverage,
            "found_keywords": found_keywords,
            "total_keywords": len(expected_keywords),
            "response_length": len(response)
        }
    
    @staticmethod
    def evaluate_retrieval(retrieved_docs: List[Dict], relevant_pmids: List[str]) -> Dict[str, float]:
        """
        Evaluate retrieval quality.
        
        Args:
            retrieved_docs: Retrieved documents
            relevant_pmids: Known relevant PMIDs
            
        Returns:
            Precision and recall metrics
        """
        retrieved_pmids = {doc['metadata'].get('pmid') for doc in retrieved_docs}
        relevant_set = set(relevant_pmids)
        
        if not relevant_set:
            return {"precision": 0, "recall": 0, "f1": 0}
        
        true_positives = len(retrieved_pmids & relevant_set)
        false_positives = len(retrieved_pmids - relevant_set)
        false_negatives = len(relevant_set - retrieved_pmids)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


# Sample test cases
TEST_CASES = [
    {
        "question": "What are the recent breakthroughs in machine learning?",
        "expected_keywords": ["machine learning", "neural networks", "model"],
        "relevant_pmids": []  # Add actual PMIDs after testing
    },
    {
        "question": "How does deep learning compare to traditional machine learning?",
        "expected_keywords": ["deep learning", "machine learning", "neural"],
        "relevant_pmids": []
    },
]


def run_evaluation(agent, test_cases: List[Dict] = None) -> None:
    """
    Run evaluation on test cases.
    
    Args:
        agent: QA agent to evaluate
        test_cases: List of test cases
    """
    if test_cases is None:
        test_cases = TEST_CASES
    
    evaluator = QAEvaluator()
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['question']}")
        
        # Get response
        response = agent.run(test['question'])
        
        # Evaluate
        qa_metrics = evaluator.evaluate_response(response, test['expected_keywords'])
        
        result = {
            "test_id": i,
            "question": test['question'],
            "metrics": qa_metrics
        }
        
        results.append(result)
        
        print(f"  Coverage: {qa_metrics['coverage']:.2%}")
        print(f"  Response length: {qa_metrics['response_length']} chars")
    
    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Evaluation complete. Results saved to evaluation_results.json")
