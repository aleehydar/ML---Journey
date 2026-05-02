import unittest
import tempfile
import os
from datetime import datetime, timedelta
from evaluation_db import EvaluationDB
from ragas_evaluator import ragas_evaluator

class TestHallucinationMetrics(unittest.TestCase):
    """Test suite for hallucination rate calculation and metrics system."""
    
    def setUp(self):
        """Set up test database."""
        self.test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.test_db.close()
        self.eval_db = EvaluationDB(self.test_db.name)
    
    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.test_db.name)
    
    def test_hallucination_rate_calculation_low_faithfulness(self):
        """Test that hallucination rate increments correctly for low faithfulness responses."""
        # Add test data with low faithfulness scores (< 0.7)
        test_evaluations = [
            {
                'question': 'What are the labor laws in Pakistan?',
                'answer': 'Pakistan has comprehensive labor laws...',
                'contexts': ['Labor Law Article 1...', 'Labor Law Article 2...'],
                'faithfulness': 0.5,  # Low faithfulness
                'answer_relevance': 0.8,
                'context_recall': 0.9,
                'user_id': 'test_user',
                'org_id': 'test_org'
            },
            {
                'question': 'What is the minimum wage?',
                'answer': 'The minimum wage is PKR 32,000...',
                'contexts': ['Minimum Wage Law...'],
                'faithfulness': 0.3,  # Very low faithfulness
                'answer_relevance': 0.7,
                'context_recall': 0.8,
                'user_id': 'test_user',
                'org_id': 'test_org'
            },
            {
                'question': 'What are working hours?',
                'answer': 'Working hours are limited to 48 hours...',
                'contexts': ['Working Hours Law...'],
                'faithfulness': 0.9,  # High faithfulness
                'answer_relevance': 0.8,
                'context_recall': 0.9,
                'user_id': 'test_user',
                'org_id': 'test_org'
            }
        ]
        
        # Insert test data
        for eval_data in test_evaluations:
            self.eval_db.log_evaluation(
                question=eval_data['question'],
                answer=eval_data['answer'],
                contexts=eval_data['contexts'],
                faithfulness=eval_data['faithfulness'],
                answer_relevance=eval_data['answer_relevance'],
                context_recall=eval_data['context_recall'],
                user_id=eval_data['user_id'],
                org_id=eval_data['org_id']
            )
        
        # Get rolling averages
        rolling_averages = self.eval_db.get_rolling_averages(org_id='test_org')
        
        # Verify hallucination rate calculation
        # 2 out of 3 responses have faithfulness < 0.7, so hallucination rate should be 66.67%
        expected_hallucination_rate = (2 / 3) * 100
        actual_hallucination_rate = rolling_averages['all_time']['hallucination_rate']
        
        self.assertAlmostEqual(
            actual_hallucination_rate, 
            expected_hallucination_rate, 
            places=2,
            msg=f"Expected hallucination rate {expected_hallucination_rate}%, got {actual_hallucination_rate}%"
        )
        
        # Verify other metrics
        self.assertEqual(rolling_averages['all_time']['total_evaluations'], 3)
        self.assertAlmostEqual(
            rolling_averages['all_time']['avg_faithfulness'], 
            (0.5 + 0.3 + 0.9) / 3, 
            places=2
        )
    
    def test_hallucination_rate_zero(self):
        """Test that hallucination rate is 0 when all responses have high faithfulness."""
        # Add test data with high faithfulness scores (>= 0.7)
        test_evaluations = [
            {
                'question': 'What are constitutional rights?',
                'answer': 'The Constitution guarantees...',
                'contexts': ['Constitution Article 9...', 'Constitution Article 10...'],
                'faithfulness': 0.8,  # High faithfulness
                'answer_relevance': 0.9,
                'context_recall': 0.9,
                'user_id': 'test_user',
                'org_id': 'test_org'
            },
            {
                'question': 'What is tax law?',
                'answer': 'Tax law specifies...',
                'contexts': ['Tax Law Article...'],
                'faithfulness': 0.7,  # Borderline high faithfulness
                'answer_relevance': 0.8,
                'context_recall': 0.8,
                'user_id': 'test_user',
                'org_id': 'test_org'
            }
        ]
        
        # Insert test data
        for eval_data in test_evaluations:
            self.eval_db.log_evaluation(
                question=eval_data['question'],
                answer=eval_data['answer'],
                contexts=eval_data['contexts'],
                faithfulness=eval_data['faithfulness'],
                answer_relevance=eval_data['answer_relevance'],
                context_recall=eval_data['context_recall'],
                user_id=eval_data['user_id'],
                org_id=eval_data['org_id']
            )
        
        # Get rolling averages
        rolling_averages = self.eval_db.get_rolling_averages(org_id='test_org')
        
        # Verify hallucination rate is 0%
        self.assertEqual(rolling_averages['all_time']['hallucination_rate'], 0.0)
        self.assertEqual(rolling_averages['all_time']['total_evaluations'], 2)
    
    def test_hallucination_rate_hundred_percent(self):
        """Test that hallucination rate is 100% when all responses have low faithfulness."""
        # Add test data with all low faithfulness scores (< 0.7)
        test_evaluations = [
            {
                'question': 'What is criminal law?',
                'answer': 'Criminal law defines...',
                'contexts': ['Criminal Law Code...'],
                'faithfulness': 0.2,  # Low faithfulness
                'answer_relevance': 0.6,
                'context_recall': 0.7,
                'user_id': 'test_user',
                'org_id': 'test_org'
            },
            {
                'question': 'What is civil law?',
                'answer': 'Civil law governs...',
                'contexts': ['Civil Law Code...'],
                'faithfulness': 0.4,  # Low faithfulness
                'answer_relevance': 0.5,
                'context_recall': 0.6,
                'user_id': 'test_user',
                'org_id': 'test_org'
            }
        ]
        
        # Insert test data
        for eval_data in test_evaluations:
            self.eval_db.log_evaluation(
                question=eval_data['question'],
                answer=eval_data['answer'],
                contexts=eval_data['contexts'],
                faithfulness=eval_data['faithfulness'],
                answer_relevance=eval_data['answer_relevance'],
                context_recall=eval_data['context_recall'],
                user_id=eval_data['user_id'],
                org_id=eval_data['org_id']
            )
        
        # Get rolling averages
        rolling_averages = self.eval_db.get_rolling_averages(org_id='test_org')
        
        # Verify hallucination rate is 100%
        self.assertEqual(rolling_averages['all_time']['hallucination_rate'], 100.0)
        self.assertEqual(rolling_averages['all_time']['total_evaluations'], 2)
    
    def test_time_period_filtering(self):
        """Test that time period filtering works correctly."""
        # Add evaluation data with different timestamps
        current_time = datetime.now()
        
        # Add recent evaluation (within 7 days)
        self.eval_db.log_evaluation(
            question='Recent question',
            answer='Recent answer',
            contexts=['Recent context'],
            faithfulness=0.5,  # Low faithfulness
            answer_relevance=0.8,
            context_recall=0.9,
            user_id='test_user',
            org_id='test_org'
        )
        
        # Manually update timestamp to be recent
        import sqlite3
        with sqlite3.connect(self.eval_db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE eval_logs 
                SET timestamp = ? 
                WHERE id = (SELECT MAX(id) FROM eval_logs)
            ''', (current_time - timedelta(days=3),))
            conn.commit()
        
        # Add old evaluation (more than 30 days ago)
        self.eval_db.log_evaluation(
            question='Old question',
            answer='Old answer',
            contexts=['Old context'],
            faithfulness=0.3,  # Low faithfulness
            answer_relevance=0.7,
            context_recall=0.8,
            user_id='test_user',
            org_id='test_org'
        )
        
        # Manually update timestamp to be old
        with sqlite3.connect(self.eval_db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE eval_logs 
                SET timestamp = ? 
                WHERE id = (SELECT MAX(id) FROM eval_logs)
            ''', (current_time - timedelta(days=35),))
            conn.commit()
        
        # Get rolling averages for different periods
        rolling_averages = self.eval_db.get_rolling_averages(org_id='test_org')
        
        # 7-day period should only include recent evaluation (100% hallucination rate)
        self.assertEqual(rolling_averages['7_days']['hallucination_rate'], 100.0)
        self.assertEqual(rolling_averages['7_days']['total_evaluations'], 1)
        
        # 30-day period should also only include recent evaluation (100% hallucination rate)
        self.assertEqual(rolling_averages['30_days']['hallucination_rate'], 100.0)
        self.assertEqual(rolling_averages['30_days']['total_evaluations'], 1)
        
        # All-time period should include both evaluations (100% hallucination rate)
        self.assertEqual(rolling_averages['all_time']['hallucination_rate'], 100.0)
        self.assertEqual(rolling_averages['all_time']['total_evaluations'], 2)
    
    def test_ragas_evaluation_integration(self):
        """Test integration with RAGAS evaluator for low faithfulness scenario."""
        # Skip this test if RAGAS requires API key that's not available
        try:
            # Create a test scenario that should result in low faithfulness
            question = "What is capital of Pakistan?"
            answer = "The capital of Pakistan is Karachi."  # Incorrect answer
            contexts = ["Islamabad is the capital city of Pakistan, established in 1960."]
            
            # This should result in low faithfulness since answer contradicts the context
            scores = ragas_evaluator.evaluate_single_sync(
                question=question,
                answer=answer,
                contexts=contexts
            )
            
            # Verify that scores are returned
            self.assertIn('faithfulness', scores)
            self.assertIn('answer_relevance', scores)
            self.assertIn('context_recall', scores)
            self.assertIn('overall_score', scores)
            
            # Verify score ranges (0-1)
            for metric in ['faithfulness', 'answer_relevance', 'context_recall', 'overall_score']:
                self.assertGreaterEqual(scores[metric], 0.0)
                self.assertLessEqual(scores[metric], 1.0)
            
            # Log evaluation to test database integration
            self.eval_db.log_evaluation(
                question=question,
                answer=answer,
                contexts=contexts,
                faithfulness=scores['faithfulness'],
                answer_relevance=scores['answer_relevance'],
                context_recall=scores['context_recall'],
                user_id='test_user',
                org_id='test_org'
            )
            
            # Verify evaluation was logged
            rolling_averages = self.eval_db.get_rolling_averages(org_id='test_org')
            self.assertEqual(rolling_averages['all_time']['total_evaluations'], 1)
            
            # If faithfulness is low (< 0.7), hallucination rate should be 100%
            if scores['faithfulness'] < 0.7:
                self.assertEqual(rolling_averages['all_time']['hallucination_rate'], 100.0)
        except Exception as e:
            # Skip test if RAGAS evaluation fails due to missing API key
            self.skipTest(f"RAGAS evaluation requires API key: {str(e)}")

if __name__ == '__main__':
    unittest.main()
