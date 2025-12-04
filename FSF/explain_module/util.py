import os
# import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score

def summarize_trial(agents):
    correct = [a for a in agents if a.is_correct()]
    incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]
    mcc_score = calculate_mcc(agents)
    return correct, incorrect, mcc_score

def calculate_metrics(agents):
    """Calculate Precision, Recall, F1 and MCC metrics"""
    y_true = []
    y_pred = []
    
    for agent in agents:
        if agent.is_finished():
            # Convert labels to binary (positive=1, negative=0)
            y_true.append(1 if agent.target.lower() == "positive" else 0)
            y_pred.append(1 if agent.prediction.lower() == "positive" else 0)
    
    if len(y_true) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'mcc': 0.0
        }
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mcc': mcc
    }

def remove_fewshot(prompt: str) -> str:
    # اگر بخش مثال‌ها وجود نداشت، پرامپت را بدون تغییر برگردان
    if 'Here are some examples:' not in prompt or '(END OF EXAMPLES)' not in prompt:
        return prompt.strip('\n').strip()
        
    # اگر بخش مثال‌ها وجود داشت، آن را حذف کن
    prefix = prompt.split('Here are some examples:')[0]
    suffix = prompt.split('(END OF EXAMPLES)')[1]
    return prefix.strip('\n').strip() + '\n\n' +  suffix.strip('\n').strip()

def remove_reflections(prompt: str) -> str:
    prefix = prompt.split('You have attempted to tackle the following task before and failed.')[0]
    suffix = prompt.split('\n\nFacts:')[-1]
    return prefix.strip('\n').strip() + '\n\nFacts' +  suffix.strip('\n').strip()

def log_trial(agents, trial_n):
    correct, incorrect, mcc_score = summarize_trial(agents)

    log = f"""
########################################
BEGIN TRIAL {trial_n}
Trial summary: Correct: {len(correct)}, Incorrect: {len(incorrect)}
Matthews Correlation Coefficient (MCC): {mcc_score:.4f}
#######################################
"""

    log += '------------- BEGIN CORRECT AGENTS -------------\n\n'
    for agent in correct:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.target}\n\n'

    log += '------------- BEGIN INCORRECT AGENTS -----------\n\n'
    for agent in incorrect:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.target}\n\n'

    return log

def save_agents(agents, dir: str):
    os.makedirs(dir, exist_ok=True)
    for i, agent in enumerate(agents):
        joblib.dump(agent, os.path.join(dir, f'{i}.joblib'))

def save_results(agents, dir: str):
    os.makedirs(dir, exist_ok=True)
    results = pd.DataFrame()
    for agent in agents:
        results = pd.concat([results, pd.DataFrame([{
                                        'Prompt': remove_fewshot(agent._build_agent_prompt()),
                                        'Response': agent.scratchpad.split('Price Movement: ')[-1],
                                        'Target': agent.target,
                                        'Prediction': agent.prediction,
                                        'Probability': agent.probability if hasattr(agent, 'probability') else 0.0,
                                        'Threshold': agent.threshold if hasattr(agent, 'threshold') else 0.7,
                                        'Threshold_Status': agent.get_threshold_status() if hasattr(agent, 'get_threshold_status') else 'UNKNOWN',
                                        'Confidence_Level': agent.get_confidence_level() if hasattr(agent, 'get_confidence_level') else 'UNKNOWN',
                                        'Is_Correct': agent.is_correct()
                                        }])], ignore_index=True)
    
    # Calculate metrics
    metrics = calculate_metrics(agents)
    mcc_score = calculate_mcc(agents)
    results.to_csv(os.path.join(dir, 'results.csv'), index=False)
    
    # Calculate threshold statistics
    accepted_correct = len([a for a in agents if hasattr(a, 'get_threshold_status') and a.get_threshold_status() == "ACCEPTED_CORRECT"])
    accepted_incorrect = len([a for a in agents if hasattr(a, 'get_threshold_status') and a.get_threshold_status() == "ACCEPTED_INCORRECT"])
    rejected_low_confidence = len([a for a in agents if hasattr(a, 'get_threshold_status') and a.get_threshold_status() == "REJECTED_LOW_CONFIDENCE"])
    total = len(agents)
    
    # Save summary metrics
    with open(os.path.join(dir, 'metrics.txt'), 'w') as f:
        f.write(f'Total Samples: {len(agents)}\n')
        f.write(f'Correct Predictions: {len([a for a in agents if a.is_correct()])}\n')
        f.write(f'Precision: {metrics["precision"]:.4f}\n')
        f.write(f'Recall: {metrics["recall"]:.4f}\n')
        f.write(f'F1-Score: {metrics["f1_score"]:.4f}\n')
        f.write(f'Matthews Correlation Coefficient (MCC): {mcc_score:.4f}\n')
        f.write(f'\nThreshold Statistics:\n')
        f.write(f'Total predictions: {total}\n')
        f.write(f'Accepted (correct): {accepted_correct} ({accepted_correct/total*100:.1f}%)\n')
        f.write(f'Accepted (incorrect): {accepted_incorrect} ({accepted_incorrect/total*100:.1f}%)\n')
        f.write(f'Rejected (low confidence): {rejected_low_confidence} ({rejected_low_confidence/total*100:.1f}%)\n')
        
        if accepted_correct + accepted_incorrect > 0:
            acceptance_accuracy = accepted_correct / (accepted_correct + accepted_incorrect)
            f.write(f'Accuracy of accepted predictions: {acceptance_accuracy:.4f}\n')

def calculate_mcc(agents):
    y_true = []
    y_pred = []
    for agent in agents:
        if agent.is_finished():
            y_true.append(1 if agent.target.lower() == "positive" else 0)
            y_pred.append(1 if agent.prediction.lower() == "positive" else 0)
    return matthews_corrcoef(y_true, y_pred) if len(y_true) > 0 else 0.0
