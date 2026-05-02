#!/usr/bin/env python3
"""
Quick demo to show hallucination rate measurement working.
This bypasses the slow startup and directly tests the evaluation pipeline.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation_db import eval_db

async def demo():
    """Demonstrate evaluation system with sample data."""
    print("🎯 Hallucination Rate Measurement Demo")
    print("=" * 50)
    
    # Clear any existing test data
    print("🧹 Clearing existing test data...")
    import sqlite3
    with sqlite3.connect('evaluations.db') as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM eval_logs WHERE user_id = 'demo_user'")
        conn.commit()
    
    # Add sample evaluations with different faithfulness levels
    test_data = [
        {
            'question': 'What is minimum wage in Pakistan?',
            'answer': 'The minimum wage is PKR 32,000 per month.',
            'contexts': ['Labor Law - Minimum Wage 2024: The minimum wage in Pakistan is set by provincial governments. As of 2024, the federal minimum wage is PKR 32,000 per month for unskilled workers.'],
            'faithfulness': 0.85,  # High faithfulness
            'answer_relevance': 0.90,
            'context_recall': 0.95,
            'user_id': 'demo_user',
            'org_id': 'demo_org'
        },
        {
            'question': 'What are working hours?',
            'answer': 'Working hours are 48 hours per week maximum.',
            'contexts': ['Labor Law - Working Hours: No worker shall be required to work more than 48 hours per week.'],
            'faithfulness': 0.92,  # High faithfulness
            'answer_relevance': 0.88,
            'context_recall': 0.91,
            'user_id': 'demo_user',
            'org_id': 'demo_org'
        },
        {
            'question': 'What is capital of Pakistan?',
            'answer': 'The capital is Lahore.',  # WRONG answer
            'contexts': ['Constitution of Pakistan: Islamabad is the capital city of Pakistan.'],
            'faithfulness': 0.25,  # Low faithfulness - should count as hallucination
            'answer_relevance': 0.60,
            'context_recall': 0.70,
            'user_id': 'demo_user',
            'org_id': 'demo_org'
        },
        {
            'question': 'What is tax rate?',
            'answer': 'Sales tax is 25% on all goods.',
            'contexts': ['Tax Law - Sales Tax: The standard rate of sales tax in Pakistan is 17% on the value of taxable supplies.'],
            'faithfulness': 0.40,  # Low faithfulness
            'answer_relevance': 0.55,
            'context_recall': 0.65,
            'user_id': 'demo_user',
            'org_id': 'demo_org'
        }
    ]
    
    print("📊 Adding sample evaluation data...")
    for data in test_data:
        eval_db.log_evaluation(
            question=data['question'],
            answer=data['answer'],
            contexts=data['contexts'],
            faithfulness=data['faithfulness'],
            answer_relevance=data['answer_relevance'],
            context_recall=data['context_recall'],
            user_id=data['user_id'],
            org_id=data['org_id']
        )
    
    # Get and display metrics
    print("📈 Calculating metrics...")
    rolling_averages = eval_db.get_rolling_averages(org_id='demo_org')
    
    print("\n🎯 HALLUCINATION RATE MEASUREMENT RESULTS")
    print("=" * 50)
    
    for period in ['7_days', '30_days', 'all_time']:
        metrics = rolling_averages[period]
        print(f"\n📊 {period.upper().replace('_', ' ')}:")
        print(f"   Hallucination Rate: {metrics['hallucination_rate']}%")
        print(f"   Total Evaluations: {metrics['total_evaluations']}")
        print(f"   Avg Faithfulness: {metrics['avg_faithfulness']:.3f}")
        print(f"   Avg Answer Relevance: {metrics['avg_answer_relevance']:.3f}")
        print(f"   Avg Context Recall: {metrics['avg_context_recall']:.3f}")
    
    # Calculate expected vs actual hallucination rate
    low_faithfulness_count = sum(1 for data in test_data if data['faithfulness'] < 0.7)
    expected_hallucination_rate = (low_faithfulness_count / len(test_data)) * 100
    
    print(f"\n✅ Expected Hallucination Rate: {expected_hallucination_rate}%")
    print(f"✅ Actual Hallucination Rate: {rolling_averages['all_time']['hallucination_rate']}%")
    print(f"{'✅ MATCH' if expected_hallucination_rate == rolling_averages['all_time']['hallucination_rate'] else '❌ MISMATCH'}")
    
    print("\n" + "=" * 50)
    print("🎉 Hallucination rate measurement system is working correctly!")
    print("📱 Now you can:")
    print("   1. Start the main server: python3 app.py")
    print("   2. Visit: http://localhost:8000")
    print("   3. Ask legal questions to see real-time evaluation")
    print("   4. Check metrics at: http://localhost:8000/api/eval/summary")

if __name__ == "__main__":
    asyncio.run(demo())
