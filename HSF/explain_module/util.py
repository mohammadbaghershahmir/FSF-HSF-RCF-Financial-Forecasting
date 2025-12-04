import os
# import joblib
import pandas as pd

def summarize_trial(agents):
    correct = [a for a in agents if a.is_correct()]
    incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]
    return correct, incorrect

def remove_fewshot(prompt: str) -> str:
    try:
        if 'Here are some examples:' in prompt and '(END OF EXAMPLES)' in prompt:
            prefix = prompt.split('Here are some examples:')[0]
            suffix = prompt.split('(END OF EXAMPLES)')[1]
            return prefix.strip('\n').strip() + '\n\n' + suffix.strip('\n').strip()
        else:
            # اگر الگوی مورد نظر وجود نداشت، کل پرامپت را برگردان
            return prompt
    except:
        # در صورت خطا، کل پرامپت را برگردان
        return prompt

def remove_reflections(prompt: str) -> str:
    try:
        if 'You have attempted to tackle the following task before and failed.' in prompt and '\n\nFacts:' in prompt:
            prefix = prompt.split('You have attempted to tackle the following task before and failed.')[0]
            suffix = prompt.split('\n\nFacts:')[-1]
            return prefix.strip('\n').strip() + '\n\nFacts' + suffix.strip('\n').strip()
        else:
            # اگر الگوی مورد نظر وجود نداشت، کل پرامپت را برگردان
            return prompt
    except:
        # در صورت خطا، کل پرامپت را برگردان
        return prompt

def log_trial(agents, trial_n):
    correct, incorrect = summarize_trial(agents)

    log = f"""
########################################
BEGIN TRIAL {trial_n}
Trial summary: Correct: {len(correct)}, Incorrect: {len(incorrect)}
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
    
    # برای محاسبه معیارهای ارزیابی
    true_labels = []
    predicted_labels = []
    
    for agent in agents:
        if agent.is_finished():
            true_label = agent.target
            predicted_label = agent.prediction
            
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
        
        results = pd.concat([results, pd.DataFrame([{
                                        'Prompt': remove_fewshot(agent._build_agent_prompt()),
                                        'Response': agent.scratchpad.split('Price Movement: ')[-1],
                                        'Target': agent.target,
                                        'Predicted': agent.prediction if agent.is_finished() else 'Unknown',
                                        'Correct': agent.is_correct() if agent.is_finished() else False
                                        }])], ignore_index=True)
    
    # محاسبه معیارهای ارزیابی
    if true_labels and predicted_labels:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
        
        # تبدیل به binary
        true_binary = [1 if label.lower() == 'positive' else 0 for label in true_labels]
        pred_binary = [1 if label.lower() == 'positive' else 0 for label in predicted_labels]
        
        accuracy = accuracy_score(true_binary, pred_binary)
        precision = precision_score(true_binary, pred_binary, zero_division=0)
        recall = recall_score(true_binary, pred_binary, zero_division=0)
        f1 = f1_score(true_binary, pred_binary, zero_division=0)
        mcc = matthews_corrcoef(true_binary, pred_binary)
        cm = confusion_matrix(true_binary, pred_binary)
        
        # ذخیره معیارها
        metrics_df = pd.DataFrame([{
            'Metric': 'Accuracy',
            'Value': accuracy
        }, {
            'Metric': 'Precision', 
            'Value': precision
        }, {
            'Metric': 'Recall',
            'Value': recall
        }, {
            'Metric': 'F1-Score',
            'Value': f1
        }, {
            'Metric': 'MCC',
            'Value': mcc
        }])
        
        metrics_df.to_csv(dir + 'evaluation_metrics.csv', index=False)
        
        # ذخیره confusion matrix
        if cm.shape == (2, 2):
            cm_df = pd.DataFrame(cm, 
                               columns=['Predicted Negative', 'Predicted Positive'],
                               index=['Actual Negative', 'Actual Positive'])
        else:
            # اگر فقط یک کلاس وجود داشته باشد
            print(f"Warning: Confusion matrix shape is {cm.shape}, only one class found")
            cm_df = pd.DataFrame(cm, 
                               columns=['Predicted'],
                               index=['Actual'])
        cm_df.to_csv(dir + 'confusion_matrix.csv')
        
        print(f"\n=== EVALUATION METRICS ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm_df)
    
    results.to_csv(dir + 'results.csv', index=False)
