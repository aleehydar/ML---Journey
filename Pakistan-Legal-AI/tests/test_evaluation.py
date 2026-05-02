#!/usr/bin/env python3
"""
Test script to verify evaluation system works without API key dependencies.
"""

import asyncio
from evaluation_db import eval_db
from ragas_evaluator import ragas_evaluator

async def test_evaluation():
    """Test evaluation with mock data."""
    print("🧪 Testing evaluation system...")
    
    # Test database logging
    print("📊 Testing database logging...")
    eval_id = eval_db.log_evaluation(
        question="What is minimum wage in Pakistan?",
        answer="The minimum wage is PKR 32,000 per month for unskilled workers.",
        contexts=["Labor Law - Minimum Wage 2024: The minimum wage in Pakistan is set by provincial governments. As of 2024, the federal minimum wage is PKR 32,000 per month for unskilled workers."],
        faithfulness=0.85,  # High faithfulness
        answer_relevance=0.90,
        context_recall=0.95,
        user_id="test_user",
        org_id="test_org"
    )
    
    print(f"✅ Evaluation logged with ID: {eval_id}")
    
    # Test rolling averages
    print("📈 Testing rolling averages...")
    rolling_averages = eval_db.get_rolling_averages(org_id="test_org")
    
    print(f"📊 7-day metrics: {rolling_averages['7_days']}")
    print(f"📊 30-day metrics: {rolling_averages['30_days']}")
    print(f"📊 All-time metrics: {rolling_averages['all_time']}")
    
    # Test time series
    print("📈 Testing time series...")
    time_series = eval_db.get_time_series_data(org_id="test_org", days=30)
    print(f"📈 Time series points: {len(time_series)}")
    
    # Add more test data to test hallucination rate calculation
    print("🧪 Adding test data for hallucination rate calculation...")
    
    # Add low faithfulness evaluation (should count as hallucination)
    eval_db.log_evaluation(
        question="What is capital of Pakistan?",
        answer="The capital is Lahore.",  # Wrong answer
        contexts=["Constitution of Pakistan: Islamabad is the capital city of Pakistan."],
        faithfulness=0.3,  # Low faithfulness
        answer_relevance=0.6,
        context_recall=0.8,
        user_id="test_user",
        org_id="test_org"
    )
    
    # Add another high faithfulness evaluation
    eval_db.log_evaluation(
        question="What are working hours?",
        answer="Working hours are limited to 48 hours per week.",
        contexts=["Labor Law - Working Hours: No worker shall be required to work more than 48 hours per week."],
        faithfulness=0.9,  # High faithfulness
        answer_relevance=0.85,
        context_recall=0.95,
        user_id="test_user",
        org_id="test_org"
    )
    
    # Check updated metrics
    updated_averages = eval_db.get_rolling_averages(org_id="test_org")
    print(f"🎯 Updated hallucination rate: {updated_averages['all_time']['hallucination_rate']}%")
    print(f"🎯 Total evaluations: {updated_averages['all_time']['total_evaluations']}")
    
    print("✅ Evaluation system test complete!")

if __name__ == "__main__":
    asyncio.run(test_evaluation())
