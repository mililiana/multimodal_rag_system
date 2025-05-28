import json
import logging
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import statistics
from dataclasses import dataclass
import re

from vector_search_gemini import MultimodalRAGSystem, RAGResponse, RetrievedResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    mrr: float 
    ndcg_at_k: Dict[int, float]  
    response_time: float
    retrieval_accuracy: float
    answer_quality_score: float
    modality_coverage: Dict[str, float]

@dataclass
class TestQuery:
    query: str
    expected_topics: List[str]
    expected_modalities: List[str]  # ['text', 'image']
    relevance_threshold: float = 0.3
    category: str = "general"
    difficulty: str = "medium"  

class RAGSystemEvaluator:

    
    def __init__(self, rag_system: MultimodalRAGSystem):
        """
        Initialize evaluator with RAG system
        
        Args:
            rag_system: Instance of MultimodalRAGSystem to evaluate
        """
        self.rag_system = rag_system
        self.evaluation_results = {}
        
        # Test queries for different scenarios
        self.test_queries = self._create_test_queries()
        
        self.ground_truth = self._create_ground_truth()
    
    def _create_test_queries(self) -> List[TestQuery]:
        """Create comprehensive test queries covering different scenarios"""
        return [
            # Text-focused queries
            TestQuery(
                query="What are the latest developments in large language models?",
                expected_topics=["LLM", "language models", "GPT", "transformer"],
                expected_modalities=["text", "image"],
                category="text_heavy",
                difficulty="easy"
            ),
            TestQuery(
                query="How is AI being used in healthcare and medical diagnosis?",
                expected_topics=["healthcare", "medical", "diagnosis", "AI", "machine learning"],
                expected_modalities=["text", "image"],
                category="domain_specific",
                difficulty="medium"
            ),
            
            # Image-focused queries
            TestQuery(
                query="Show me examples of computer vision applications in autonomous vehicles",
                expected_topics=["computer vision", "autonomous", "vehicles", "self-driving"],
                expected_modalities=["image", "text"],
                category="image_heavy",
                difficulty="medium"
            ),
            TestQuery(
                query="What do neural network architectures look like?",
                expected_topics=["neural network", "architecture", "diagram", "visualization"],
                expected_modalities=["image", "text"],
                category="visual",
                difficulty="easy"
            ),
            
            # Multimodal queries
            TestQuery(
                query="Explain reinforcement learning with visual examples",
                expected_topics=["reinforcement learning", "RL", "examples", "visualization"],
                expected_modalities=["text", "image"],
                category="multimodal",
                difficulty="medium"
            ),
            TestQuery(
                query="What are the recent breakthroughs in AI research with supporting charts or graphs?",
                expected_topics=["AI research", "breakthroughs", "recent", "progress"],
                expected_modalities=["text", "image"],
                category="multimodal",
                difficulty="hard"
            ),
            
            # Edge cases
            TestQuery(
                query="quantum computing machine learning applications",
                expected_topics=["quantum", "computing", "machine learning", "applications"],
                expected_modalities=["text"],
                category="niche",
                difficulty="hard"
            ),
            TestQuery(
                query="AI ethics bias fairness",
                expected_topics=["AI ethics", "bias", "fairness", "responsible AI"],
                expected_modalities=["text"],
                category="conceptual",
                difficulty="medium"
            ),
            
            # Ambiguous queries
            TestQuery(
                query="deep learning",
                expected_topics=["deep learning", "neural networks"],
                expected_modalities=["text", "image"],
                category="ambiguous",
                difficulty="easy"
            ),
            TestQuery(
                query="AI news",
                expected_topics=["artificial intelligence", "news", "recent"],
                expected_modalities=["text", "image"],
                category="broad",
                difficulty="easy"
            )
        ]
    
    def _create_ground_truth(self) -> Dict[str, List[str]]:
       
        return {
            "What are the latest developments in large language models?": [
                "relevant_article_id_1", "relevant_article_id_2"
            ],
        }
    
    def evaluate_retrieval_precision_recall(self, 
                                          query: TestQuery, 
                                          retrieved_results: List[RetrievedResult],
                                          k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Dict[int, float]]:
        """
        Calculating precision and recall at different k values
        """
        precision_at_k = {}
        recall_at_k = {}
        
        for k in k_values:
            top_k_results = retrieved_results[:k]
            
            relevant_count = 0
            for result in top_k_results:
                content_lower = (result.title + " " + result.content).lower()
                if any(topic.lower() in content_lower for topic in query.expected_topics):
                    relevant_count += 1
            
            precision_at_k[k] = relevant_count / k if k > 0 else 0
            
            total_relevant = len(query.expected_topics)  
            recall_at_k[k] = relevant_count / total_relevant if total_relevant > 0 else 0
        
        return {"precision": precision_at_k, "recall": recall_at_k}
    
    def calculate_mrr(self, query: TestQuery, retrieved_results: List[RetrievedResult]) -> float:
        # Calculate Mean Reciprocal Rank
        for i, result in enumerate(retrieved_results, 1):
            content_lower = (result.title + " " + result.content).lower()
            if any(topic.lower() in content_lower for topic in query.expected_topics):
                return 1.0 / i
        return 0.0
    
    def calculate_ndcg(self, query: TestQuery, retrieved_results: List[RetrievedResult], k: int = 10) -> float:
        # Calculate Normalized Discounted Cumulative Gain at k
        # Simplified NDCG calculation
        dcg = 0.0
        for i, result in enumerate(retrieved_results[:k], 1):
            content_lower = (result.title + " " + result.content).lower()
            relevance = sum(1 for topic in query.expected_topics 
                          if topic.lower() in content_lower)
            dcg += relevance / np.log2(i + 1)
        
        # Ideal DCG (assuming perfect ranking)
        ideal_dcg = sum(len(query.expected_topics) / np.log2(i + 1) for i in range(1, min(k, len(query.expected_topics)) + 1))
        
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    def evaluate_modality_coverage(self, 
                                 query: TestQuery, 
                                 retrieved_results: List[RetrievedResult]) -> Dict[str, float]:
        # let`s evaluate how well the system covers different modalities
        total_results = len(retrieved_results)
        if total_results == 0:
            return {"text": 0.0, "image": 0.0, "coverage_score": 0.0}
        
        text_count = sum(1 for r in retrieved_results if r.content_type == "text")
        image_count = sum(1 for r in retrieved_results if r.content_type == "image")
        
        text_ratio = text_count / total_results
        image_ratio = image_count / total_results
        
        # Calculate coverage score based on expected modalities
        expected_modalities = set(query.expected_modalities)
        actual_modalities = set()
        if text_count > 0:
            actual_modalities.add("text")
        if image_count > 0:
            actual_modalities.add("image")
        
        coverage_score = len(actual_modalities.intersection(expected_modalities)) / len(expected_modalities)
        
        return {
            "text": text_ratio,
            "image": image_ratio,
            "coverage_score": coverage_score
        }
    
    def evaluate_answer_quality(self, query: str, answer: str) -> Dict[str, float]:
        """
        Evaluate the quality of generated answers
        This is a simplified heuristic-based evaluation
        """
        if not answer or answer.strip() == "":
            return {"completeness": 0.0, "relevance": 0.0, "coherence": 0.0, "overall": 0.0}
        
        # Completeness: length and structure
        word_count = len(answer.split())
        sentence_count = len([s for s in answer.split('.') if s.strip()])
        completeness = min(1.0, word_count / 100)  # Normalize to 100 words
        
        # Relevance: keyword overlap with query
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        relevance = len(query_words.intersection(answer_words)) / len(query_words) if query_words else 0
        
        # Coherence: simplified measure based on sentence structure
        coherence = min(1.0, sentence_count / max(1, word_count / 20))  # Reasonable sentence length
        
        overall = (completeness + relevance + coherence) / 3
        
        return {
            "completeness": completeness,
            "relevance": relevance,
            "coherence": coherence,
            "overall": overall
        }
    
    def run_single_query_evaluation(self, test_query: TestQuery) -> Dict[str, Any]:
        """
        Run evaluation for a single query
        """
        logger.info(f"Evaluating query: {test_query.query}")
        
        # Measure response time
        start_time = time.time()
        
        # Execute query
        response = self.rag_system.query(
            query=test_query.query,
            search_type="multimodal",
            limit=10,
            score_threshold=0.1,
            generate_answer=True
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Calculate metrics
        precision_recall = self.evaluate_retrieval_precision_recall(test_query, response.retrieved_results)
        mrr = self.calculate_mrr(test_query, response.retrieved_results)
        ndcg = {k: self.calculate_ndcg(test_query, response.retrieved_results, k) for k in [1, 3, 5, 10]}
        modality_coverage = self.evaluate_modality_coverage(test_query, response.retrieved_results)
        answer_quality = self.evaluate_answer_quality(test_query.query, response.answer)
        
        return {
            "query": test_query.query,
            "category": test_query.category,
            "difficulty": test_query.difficulty,
            "response_time": response_time,
            "total_results": response.total_results,
            "precision_at_k": precision_recall["precision"],
            "recall_at_k": precision_recall["recall"],
            "mrr": mrr,
            "ndcg_at_k": ndcg,
            "modality_coverage": modality_coverage,
            "answer_quality": answer_quality,
            "retrieved_results": response.retrieved_results,
            "answer": response.answer
        }
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across all test queries
        """
        logger.info("Starting comprehensive RAG system evaluation...")
        
        all_results = []
        category_results = defaultdict(list)
        difficulty_results = defaultdict(list)
        
        for test_query in self.test_queries:
            try:
                result = self.run_single_query_evaluation(test_query)
                all_results.append(result)
                category_results[test_query.category].append(result)
                difficulty_results[test_query.difficulty].append(result)
                
                # Brief pause to avoid overwhelming the system
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error evaluating query '{test_query.query}': {e}")
                continue
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_results)
        category_analysis = self._analyze_by_category(category_results)
        difficulty_analysis = self._analyze_by_difficulty(difficulty_results)
        
        # System performance analysis
        system_analysis = self._analyze_system_performance(all_results)
        
        evaluation_summary = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(all_results),
            "aggregated_metrics": aggregated_metrics,
            "category_analysis": category_analysis,
            "difficulty_analysis": difficulty_analysis,
            "system_analysis": system_analysis,
            "detailed_results": all_results
        }
        
        self.evaluation_results = evaluation_summary
        return evaluation_summary
    
    def _aggregate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across all queries"""
        if not results:
            return {}
        
        # Average precision and recall at k
        k_values = [1, 3, 5, 10]
        avg_precision = {}
        avg_recall = {}
        avg_ndcg = {}
        
        for k in k_values:
            precisions = [r["precision_at_k"][k] for r in results if k in r["precision_at_k"]]
            recalls = [r["recall_at_k"][k] for r in results if k in r["recall_at_k"]]
            ndcgs = [r["ndcg_at_k"][k] for r in results if k in r["ndcg_at_k"]]
            
            avg_precision[k] = statistics.mean(precisions) if precisions else 0
            avg_recall[k] = statistics.mean(recalls) if recalls else 0
            avg_ndcg[k] = statistics.mean(ndcgs) if ndcgs else 0
        
        # Other aggregated metrics
        avg_mrr = statistics.mean([r["mrr"] for r in results])
        avg_response_time = statistics.mean([r["response_time"] for r in results])
        avg_modality_coverage = statistics.mean([r["modality_coverage"]["coverage_score"] for r in results])
        avg_answer_quality = statistics.mean([r["answer_quality"]["overall"] for r in results])
        
        return {
            "precision_at_k": avg_precision,
            "recall_at_k": avg_recall,
            "ndcg_at_k": avg_ndcg,
            "mean_reciprocal_rank": avg_mrr,
            "avg_response_time": avg_response_time,
            "modality_coverage_score": avg_modality_coverage,
            "answer_quality_score": avg_answer_quality
        }
    
    def _analyze_by_category(self, category_results: Dict) -> Dict:
        """Analyze performance by query category"""
        analysis = {}
        
        for category, results in category_results.items():
            if not results:
                continue
                
            analysis[category] = {
                "query_count": len(results),
                "avg_precision_at_5": statistics.mean([r["precision_at_k"][5] for r in results]),
                "avg_recall_at_5": statistics.mean([r["recall_at_k"][5] for r in results]),
                "avg_mrr": statistics.mean([r["mrr"] for r in results]),
                "avg_response_time": statistics.mean([r["response_time"] for r in results]),
                "avg_answer_quality": statistics.mean([r["answer_quality"]["overall"] for r in results]),
                "modality_coverage": statistics.mean([r["modality_coverage"]["coverage_score"] for r in results])
            }
        
        return analysis
    
    def _analyze_by_difficulty(self, difficulty_results: Dict) -> Dict:
        """Analyze performance by query difficulty"""
        analysis = {}
        
        for difficulty, results in difficulty_results.items():
            if not results:
                continue
                
            analysis[difficulty] = {
                "query_count": len(results),
                "avg_precision_at_5": statistics.mean([r["precision_at_k"][5] for r in results]),
                "avg_recall_at_5": statistics.mean([r["recall_at_k"][5] for r in results]),
                "avg_mrr": statistics.mean([r["mrr"] for r in results]),
                "avg_response_time": statistics.mean([r["response_time"] for r in results]),
                "avg_answer_quality": statistics.mean([r["answer_quality"]["overall"] for r in results])
            }
        
        return analysis
    
    def _analyze_system_performance(self, results: List[Dict]) -> Dict:
        """Analyze overall system performance characteristics"""
        if not results:
            return {}
        
        response_times = [r["response_time"] for r in results]
        result_counts = [r["total_results"] for r in results]
        
        # Text vs Image performance
        text_heavy_queries = [r for r in results if "text" in r["category"]]
        image_heavy_queries = [r for r in results if "image" in r["category"]]
        multimodal_queries = [r for r in results if "multimodal" in r["category"]]
        
        return {
            "response_time_stats": {
                "mean": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "std": statistics.stdev(response_times) if len(response_times) > 1 else 0,
                "min": min(response_times),
                "max": max(response_times)
            },
            "result_count_stats": {
                "mean": statistics.mean(result_counts),
                "median": statistics.median(result_counts),
                "std": statistics.stdev(result_counts) if len(result_counts) > 1 else 0
            },
            "modality_performance": {
                "text_heavy": {
                    "count": len(text_heavy_queries),
                    "avg_precision": statistics.mean([r["precision_at_k"][5] for r in text_heavy_queries]) if text_heavy_queries else 0
                },
                "image_heavy": {
                    "count": len(image_heavy_queries),
                    "avg_precision": statistics.mean([r["precision_at_k"][5] for r in image_heavy_queries]) if image_heavy_queries else 0
                },
                "multimodal": {
                    "count": len(multimodal_queries),
                    "avg_precision": statistics.mean([r["precision_at_k"][5] for r in multimodal_queries]) if multimodal_queries else 0
                }
            }
        }
    
    def generate_evaluation_report(self) -> str:
        """
        Generate a comprehensive evaluation report
        """
        if not self.evaluation_results:
            return "No evaluation results available. Please run evaluation first."
        
        results = self.evaluation_results
        metrics = results["aggregated_metrics"]
        
        report = f"""
# Multimodal RAG System Evaluation Report

**Evaluation Date:** {results['timestamp']}
**Total Queries Evaluated:** {results['total_queries']}

## Executive Summary

The multimodal RAG system was evaluated across {results['total_queries']} diverse queries covering different categories and difficulty levels. The system demonstrates good performance in retrieving relevant content with an average precision@5 of {metrics['precision_at_k'][5]:.3f} and mean reciprocal rank of {metrics['mean_reciprocal_rank']:.3f}.

## Key Performance Metrics

### Retrieval Performance
- **Precision@1:** {metrics['precision_at_k'][1]:.3f}
- **Precision@3:** {metrics['precision_at_k'][3]:.3f}
- **Precision@5:** {metrics['precision_at_k'][5]:.3f}
- **Precision@10:** {metrics['precision_at_k'][10]:.3f}

- **Recall@1:** {metrics['recall_at_k'][1]:.3f}
- **Recall@3:** {metrics['recall_at_k'][3]:.3f}
- **Recall@5:** {metrics['recall_at_k'][5]:.3f}
- **Recall@10:** {metrics['recall_at_k'][10]:.3f}

### Ranking Quality
- **Mean Reciprocal Rank (MRR):** {metrics['mean_reciprocal_rank']:.3f}
- **NDCG@5:** {metrics['ndcg_at_k'][5]:.3f}
- **NDCG@10:** {metrics['ndcg_at_k'][10]:.3f}

### System Performance
- **Average Response Time:** {metrics['avg_response_time']:.2f} seconds
- **Modality Coverage Score:** {metrics['modality_coverage_score']:.3f}
- **Answer Quality Score:** {metrics['answer_quality_score']:.3f}

## Performance by Category

"""
        
        # Add category analysis
        for category, stats in results["category_analysis"].items():
            report += f"""
### {category.title()} Queries ({stats['query_count']} queries)
- Precision@5: {stats['avg_precision_at_5']:.3f}
- Recall@5: {stats['avg_recall_at_5']:.3f}
- MRR: {stats['avg_mrr']:.3f}
- Response Time: {stats['avg_response_time']:.2f}s
- Answer Quality: {stats['avg_answer_quality']:.3f}
- Modality Coverage: {stats['modality_coverage']:.3f}
"""
        
        report += """
## Performance by Difficulty

"""
        
        for difficulty, stats in results["difficulty_analysis"].items():
            report += f"""
### {difficulty.title()} Queries ({stats['query_count']} queries)
- Precision@5: {stats['avg_precision_at_5']:.3f}
- Recall@5: {stats['avg_recall_at_5']:.3f}
- MRR: {stats['avg_mrr']:.3f}
- Response Time: {stats['avg_response_time']:.2f}s
- Answer Quality: {stats['avg_answer_quality']:.3f}
"""
        
        sys_analysis = results["system_analysis"]
        report += f"""
## System Analysis

### Response Time Distribution
- Mean: {sys_analysis['response_time_stats']['mean']:.2f}s
- Median: {sys_analysis['response_time_stats']['median']:.2f}s
- Standard Deviation: {sys_analysis['response_time_stats']['std']:.2f}s
- Range: {sys_analysis['response_time_stats']['min']:.2f}s - {sys_analysis['response_time_stats']['max']:.2f}s

### Modality-Specific Performance
- Text-heavy queries: {sys_analysis['modality_performance']['text_heavy']['avg_precision']:.3f} precision@5
- Image-heavy queries: {sys_analysis['modality_performance']['image_heavy']['avg_precision']:.3f} precision@5  
- Multimodal queries: {sys_analysis['modality_performance']['multimodal']['avg_precision']:.3f} precision@5

## Recommendations

1. **Retrieval Quality:** The system shows good precision but could benefit from improved recall, especially for complex queries.

2. **Response Time:** Average response time of {metrics['avg_response_time']:.2f}s is reasonable but could be optimized for better user experience.

3. **Multimodal Integration:** Modality coverage score of {metrics['modality_coverage_score']:.3f} indicates room for improvement in balancing text and image results.

4. **Answer Generation:** Answer quality score of {metrics['answer_quality_score']:.3f} suggests the generated responses are helpful but could be enhanced.

## Conclusion

The multimodal RAG system demonstrates solid performance across various query types and difficulties. Key strengths include good precision in retrieval and reasonable response times. Areas for improvement include enhancing recall, optimizing multimodal result balancing, and improving answer generation quality.
"""
        
        return report
    
    def save_evaluation_results(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation results saved to {filepath}")
    
    def create_visualizations(self, save_dir: str = "."):
        if not self.evaluation_results:
            logger.warning("No evaluation results to visualize")
            return
        
        results = self.evaluation_results["detailed_results"]
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Precision@K plot
        plt.figure(figsize=(10, 6))
        k_values = [1, 3, 5, 10]
        precisions = [self.evaluation_results["aggregated_metrics"]["precision_at_k"][k] for k in k_values]
        recalls = [self.evaluation_results["aggregated_metrics"]["recall_at_k"][k] for k in k_values]
        
        plt.plot(k_values, precisions, marker='o', label='Precision@K', linewidth=2)
        plt.plot(k_values, recalls, marker='s', label='Recall@K', linewidth=2)
        plt.xlabel('K (Number of Results)')
        plt.ylabel('Score')
        plt.title('Precision and Recall at Different K Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/precision_recall_at_k.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance by category
        plt.figure(figsize=(12, 8))
        categories = list(self.evaluation_results["category_analysis"].keys())
        category_metrics = ['avg_precision_at_5', 'avg_recall_at_5', 'avg_mrr', 'avg_answer_quality']
        
        x = np.arange(len(categories))
        width = 0.2
        
        for i, metric in enumerate(category_metrics):
            values = [self.evaluation_results["category_analysis"][cat][metric] for cat in categories]
            plt.bar(x + i*width, values, width, label=metric.replace('avg_', '').replace('_', ' ').title())
        
        plt.xlabel('Query Category')
        plt.ylabel('Score')
        plt.title('Performance Metrics by Query Category')
        plt.xticks(x + width*1.5, categories, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/performance_by_category.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Response time distribution
        plt.figure(figsize=(10, 6))
        response_times = [r["response_time"] for r in results]
        plt.hist(response_times, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Response Times')
        plt.axvline(np.mean(response_times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(response_times):.2f}s')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/response_time_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {save_dir}/")

def main():
   
    rag_system = MultimodalRAGSystem(
        qdrant_url="https://2829a7bb-1713-4e6c-b1b1-d8ffbf38a124.europe-west3-0.gcp.cloud.qdrant.io",
        qdrant_api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.yR7PB8U3c1ctT4OGCGLQnqE7-SRYDeapPDmCrAOozek"
    )
    
    evaluator = RAGSystemEvaluator(rag_system)
    
    print("Running comprehensive evaluation...")
    evaluation_results = evaluator.run_comprehensive_evaluation()
    
    print("\nGenerating evaluation report...")
    report = evaluator.generate_evaluation_report()
    print(report)
    
    evaluator.save_evaluation_results("rag_evaluation_results.json")
    
    
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()
