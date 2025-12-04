from data_load.dataloader import DataLoader
from explain_module.util import summarize_trial, remove_reflections, save_results, calculate_metrics
from explain_module.agents import PredictReflectAgent
from predict_module.merge_peft_adapter import merge_peft_adapter
from predict_module.supervised_finetune import supervised_finetune
from predict_module.tuning_lm_with_rl import tuning_lm_with_dpo
from transformers import LlamaTokenizer, pipeline
from trl import AutoModelForCausalLMWithValueHead
import os, json


class Exp_Model:
    def __init__(self, args):
        self.args = args
        self.dataloader = DataLoader(args)

    def train(self):
        # Collect demonstration data
        print("Loading Train Agents...")
        data = self.dataloader.load(flag="train")

        agent_cls = PredictReflectAgent
        agents = [agent_cls(row['ticker'], 
                          row['market_data'], 
                          row['target'],
                          row['price_change'],
                          threshold=self.args.threshold) for _, row in data.iterrows()]
        print("Loaded Train Agents.")

        # Dictionary to track reflection attempts for each agent
        reflection_counts = {i: 0 for i in range(len(agents))}
        unsuccessful_phases = {i: 0 for i in range(len(agents))}

        for agent in agents:
            agent.run()

            if agent.is_correct():
                prompt = agent._build_agent_prompt()
                response = agent.scratchpad.split('Price Movement: ')[-1]

                sample = {"instruction": prompt, "input": "", "output": response}
                with open(self.args.data_path, 'a') as f:
                    f.write(json.dumps(sample) + "\n")

        # Calculate metrics for initial trial
        correct, incorrect, mcc_score = summarize_trial(agents)
        metrics = calculate_metrics(agents)
        total = len(correct) + len(incorrect)
        accuracy = (len(correct) / total) * 100 if total > 0 else 0
        print(f'Finished Trial 0:')
        print(f'  Correct: {len(correct)}, Incorrect: {len(incorrect)}')
        print(f'  Accuracy: {accuracy:.2f}%')
        print(f'  Precision: {metrics["precision"]:.4f}')
        print(f'  Recall: {metrics["recall"]:.4f}')
        print(f'  F1-Score: {metrics["f1_score"]:.4f}')
        print(f'  MCC: {mcc_score:.4f}')

        # Train supervised policy
        supervised_finetune(self.args)
        merge_peft_adapter(model_name=self.args.output_path, output_name=self.args.rl_base_model)

        # Collect comparison data
        comparison_data = []
        total_reflections = 0
        
        for trial in range(self.args.num_reflect_trials):
            for idx, agent in enumerate([a for a in agents if not a.is_correct()]):
                agent_idx = agents.index(agent)  # Get original index of the agent
                reflection_counts[agent_idx] += 1
                
                prev_response = agent.scratchpad.split('Price Movement: ')[-1]
                prev_prediction = agent.prediction
                prev_probability = agent.probability
                
                agent.run()
                total_reflections += 1
                
                # اگر پیش‌بینی یا احتمال تغییر نکرده، یک فاز ناموفق است
                if agent.prediction == prev_prediction and agent.probability == prev_probability:
                    unsuccessful_phases[agent_idx] += 1
                
                if agent.is_correct():
                    print(agent._build_agent_prompt(), "\n\n\n")
                    prompt = remove_reflections(agent._build_agent_prompt())
                    response = agent.scratchpad.split('Price Movement: ')[-1]

                    sample = {
                        "prompt": prompt,
                        "rejected": prev_response,
                        "chosen": response,
                        "preferred": "chosen"
                    }
                    comparison_data.append(sample)

            # Calculate metrics for each trial
            correct, incorrect, mcc_score = summarize_trial(agents)
            metrics = calculate_metrics(agents)
            total = len(correct) + len(incorrect)
            accuracy = (len(correct) / total) * 100 if total > 0 else 0
            print(f'Finished Trial {trial+1}:')
            print(f'  Correct: {len(correct)}, Incorrect: {len(incorrect)}')
            print(f'  Accuracy: {accuracy:.2f}%')
            print(f'  Precision: {metrics["precision"]:.4f}')
            print(f'  Recall: {metrics["recall"]:.4f}')
            print(f'  F1-Score: {metrics["f1_score"]:.4f}')
            print(f'  MCC: {mcc_score:.4f}')

        # چاپ آمار بازتاب‌ها برای هر نمونه
        print("\nReflection Statistics for Each Sample:")
        print("=====================================")
        for idx in range(len(agents)):
            if reflection_counts[idx] > 0:
                agent = agents[idx]
                print(f"\nSample {idx + 1}:")
                print(f"Market: {agent.market}")
                print(f"Target: {agent.target}")
                print(f"Total Reflection Attempts: {reflection_counts[idx]}")
                print(f"Unsuccessful Phases: {unsuccessful_phases[idx]}")
                print(f"Success Rate: {((reflection_counts[idx] - unsuccessful_phases[idx]) / reflection_counts[idx] * 100):.2f}% of reflection phases led to changes")
                print("-------------------------------------")

        print(f"\nTotal reflections performed across all samples: {total_reflections}")
        
        # Calculate average unsuccessful phases only if there were reflections
        samples_with_reflections = [v for v in reflection_counts.values() if v > 0]
        if samples_with_reflections:
            avg_unsuccessful = sum(unsuccessful_phases.values()) / len(samples_with_reflections)
            print(f"Average unsuccessful phases per sample: {avg_unsuccessful:.2f}")
        else:
            print("No reflections were performed during training.")

        # Threshold analysis
        print("\nThreshold Performance Analysis:")
        print("===============================")
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for threshold in thresholds:
            filtered_agents = [a for a in agents if a.is_finished() and a.probability >= threshold]
            if len(filtered_agents) > 0:
                correct = len([a for a in filtered_agents if a.is_correct()])
                accuracy = correct / len(filtered_agents)
                coverage = len(filtered_agents) / len([a for a in agents if a.is_finished()])
                print(f"Threshold {threshold:.1f}: Accuracy={accuracy:.4f}, Coverage={coverage:.4f}, Predictions={len(filtered_agents)}")
            else:
                print(f"Threshold {threshold:.1f}: No predictions meet threshold")

        os.makedirs(self.args.datasets_dir, exist_ok=True)
        comparison_data_path = os.path.join(self.args.datasets_dir, "comparison_data.json")

        if comparison_data:
            with open(comparison_data_path, 'w') as f:
                for sample in comparison_data:
                    f.write(json.dumps(sample) + "\n")

        print(f"Comparison data saved at {comparison_data_path}.")
        
        # Optimize using DPO
        tuning_lm_with_dpo(self.args)
        merge_peft_adapter(model_name=self.args.output_dir+"dpo_model", output_name="./saved_models/sep_model")

    def test(self):
        # Load test data
        print("Loading Test Agents...")
        data = self.dataloader.load(flag="test")
        
        # Initialize agents with multi-market data
        agent_cls = PredictReflectAgent
        agents = [agent_cls(row['ticker'], 
                          row['market_data'], 
                          row['target'],
                          row['price_change'],
                          threshold=self.args.threshold) for _, row in data.iterrows()]
        print("Loaded Test Agents.")

        # Test predictions
        for agent in agents:
            agent.run()

        # Calculate metrics
        correct, incorrect, mcc_score = summarize_trial(agents)
        metrics = calculate_metrics(agents)
        total = len(correct) + len(incorrect)
        accuracy = (len(correct) / total) * 100 if total > 0 else 0
        
        print(f'\nTest Results:')
        print(f'Correct: {len(correct)}')
        print(f'Incorrect: {len(incorrect)}')
        print(f'Accuracy: {accuracy:.2f}%')
        print(f'Precision: {metrics["precision"]:.4f}')
        print(f'Recall: {metrics["recall"]:.4f}')
        print(f'F1-Score: {metrics["f1_score"]:.4f}')
        print(f'Matthews Correlation Coefficient: {mcc_score:.4f}')
        
        print('\nDetailed Results:')
        print('=' * 100)
        print(f'{"Market":<8} {"Target":<10} {"Prediction":<10} {"Status":<20} {"Price Change":<12} {"Prob.":<8} {"Threshold":<10} {"Confidence":<10}')
        print('-' * 100)
        for agent in agents:
            status = agent.get_threshold_status()
            confidence_level = agent.get_confidence_level()
            print(f'{agent.market:<8} {agent.target:<10} {agent.prediction:<10} {status:<20} {agent.price_change:>+.2f}%{agent.probability:>8.2f}{agent.threshold:>10.2f}{confidence_level:>10}')
        print('=' * 100)

        print('\nIncorrect Predictions Detail:')
        print('=' * 80)
        for agent in agents:
            if not agent.is_correct():
                print(f'\nMarket: {agent.market}')
                print(f'Expected (Target): {agent.target}')
                print(f'Predicted: {agent.prediction}')
                print(f'Price Change: {agent.price_change:>+.2f}%')
                print(f'Confidence: {agent.probability:.2f}')
                print(f'Threshold: {agent.threshold:.2f}')
                print('-' * 80)

        # Threshold analysis for test set
        print("\nTest Set Threshold Performance Analysis:")
        print("========================================")
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for threshold in thresholds:
            filtered_agents = [a for a in agents if a.is_finished() and a.probability >= threshold]
            if len(filtered_agents) > 0:
                correct = len([a for a in filtered_agents if a.is_correct()])
                accuracy = correct / len(filtered_agents)
                coverage = len(filtered_agents) / len([a for a in agents if a.is_finished()])
                print(f"Threshold {threshold:.1f}: Accuracy={accuracy:.4f}, Coverage={coverage:.4f}, Predictions={len(filtered_agents)}")
            else:
                print(f"Threshold {threshold:.1f}: No predictions meet threshold")
        
        # Threshold statistics
        print("\nThreshold Decision Statistics:")
        print("==============================")
        accepted_correct = len([a for a in agents if a.get_threshold_status() == "ACCEPTED_CORRECT"])
        accepted_incorrect = len([a for a in agents if a.get_threshold_status() == "ACCEPTED_INCORRECT"])
        rejected_low_confidence = len([a for a in agents if a.get_threshold_status() == "REJECTED_LOW_CONFIDENCE"])
        total = len(agents)
        
        print(f"Total predictions: {total}")
        print(f"Accepted (correct): {accepted_correct} ({accepted_correct/total*100:.1f}%)")
        print(f"Accepted (incorrect): {accepted_incorrect} ({accepted_incorrect/total*100:.1f}%)")
        print(f"Rejected (low confidence): {rejected_low_confidence} ({rejected_low_confidence/total*100:.1f}%)")
        
        if accepted_correct + accepted_incorrect > 0:
            acceptance_accuracy = accepted_correct / (accepted_correct + accepted_incorrect)
            print(f"Accuracy of accepted predictions: {acceptance_accuracy:.4f}")
        
        # Confidence level distribution
        print("\nConfidence Level Distribution:")
        print("==============================")
        high_conf = len([a for a in agents if a.get_confidence_level() == "High"])
        medium_conf = len([a for a in agents if a.get_confidence_level() == "Medium"])
        low_conf = len([a for a in agents if a.get_confidence_level() == "Low"])
        
        print(f"High confidence: {high_conf} ({high_conf/total*100:.1f}%)")
        print(f"Medium confidence: {medium_conf} ({medium_conf/total*100:.1f}%)")
        print(f"Low confidence: {low_conf} ({low_conf/total*100:.1f}%)")

        # Save detailed results
        os.makedirs(self.args.save_dir, exist_ok=True)
        save_results(agents, self.args.save_dir)

