from data_load.dataloader import DataLoader
from explain_module.util import summarize_trial, remove_reflections, save_results
from explain_module.dual_agents import DualPredictionAgent
from predict_module.merge_peft_adapter import merge_peft_adapter
from predict_module.supervised_finetune import supervised_finetune
from predict_module.tuning_lm_with_rl import tuning_lm_with_dpo
from transformers import LlamaTokenizer, pipeline
from trl import AutoModelForCausalLMWithValueHead
from utils.evaluation_metrics import EvaluationMetrics
import os, json


class Exp_Model:
    def __init__(self, args):
        self.args = args
        self.dataloader = DataLoader(args)
        self.dual_agent = DualPredictionAgent()  # Training mode

    def train(self):
        # Collect demonstration data using dual model architecture
        print("Loading Train Data...")
        data = self.dataloader.load(flag="train")
        
        # Debug information
        print(f"Data type: {type(data)}")
        print(f"Data shape: {data.shape if hasattr(data, 'shape') else 'No shape'}")
        print(f"Data columns: {list(data.columns) if hasattr(data, 'columns') else 'No columns'}")
        
        if len(data) == 0:
            print("ERROR: No data loaded!")
            return
        
        print(f"First row: {data.iloc[0].to_dict()}")

        # Initialize data collection
        supervised_samples = []
        dpo_samples = []
        
        print("Starting dual model predictions...")
        
        for idx, row in data.iterrows():
            print(f"\n{'='*100}")
            print(f"📅 PROCESSING SAMPLE {idx + 1}/{len(data)} - DATE: {row['date']}")
            print(f"{'='*100}")
            
            # Skip if essential fields missing
            if not row.get('market_data'):
                print("Missing market_data → passing empty structures to model")
                row['market_data'] = {}
            
            # Build tweets-based summary (optional)
            btc_texts = self.dataloader.get_tweets_texts(row['date'], self.dataloader.main_market)
            gold_texts = []
            for m in self.dataloader.secondary_markets:
                gold_texts.extend(self.dataloader.get_tweets_texts(row['date'], m))
            all_texts = [t for t in (btc_texts + gold_texts) if t]
            news_summary = ""
            if all_texts:
                raw_summary = self.dataloader.summarizer.get_summary(self.dataloader.main_market, all_texts)
                if raw_summary and self.dataloader.summarizer.is_informative(raw_summary):
                    news_summary = raw_summary
            else:
                print("No tweets available → using empty summary")
            
            # Get predictions from both models
            result = self.dual_agent.predict(
                market_data=row['market_data'],
                sentiment_data={
                    'btc_sentiment': row['market_data'].get('btc_sentiment', {}),
                    'gold_sentiment': row['market_data'].get('gold_sentiment', {})
                },
                news_summary=news_summary,
                target=row.get('target') or None
            )
            
            # Safety: require a parsed prediction only
            if not result.get('final_prediction') or result['final_prediction'] not in ("Positive", "Negative", "Unknown"):
                print("Skipping: unparsable model result")
                continue
            
            if result['dataset_type'] == 'supervised' and (result.get('candle_correct', False) or result.get('news_correct', False)) and row.get('target'):
                # Enrich SFT input with structured data
                supervised_sample = {
                    "instruction": "Analyze Bitcoin and Gold market data to predict Bitcoin's price movement.",
                    "input": json.dumps({
                        "date": row['date'],
                        "market_data": row['market_data'],
                        "news_summary": news_summary
                    }, ensure_ascii=False),
                    "output": f"Bitcoin Price Movement: {result['final_prediction']}\n\nExplanation: {result['final_explanation']}",
                    "prediction": result['final_prediction']
                }
                supervised_samples.append(supervised_sample)
                print(f"✅ Sample {idx + 1}: Added to SUPERVISED dataset")
            else:
                # Collect DPO pairs only if target exists (to know correct label)
                dpo_list = []
                if row.get('target'):
                    dpo_list = self.dual_agent.create_dpo_sample(result)
                    for s in dpo_list:
                        if all(k in s for k in ("prompt","chosen","rejected")) and s["chosen"] != s["rejected"]:
                            dpo_samples.append(s)
                print(f"🔄 Sample {idx + 1}: Added {len(dpo_list)} DPO samples")

        # Save supervised learning dataset
        print(f"Saving {len(supervised_samples)} supervised samples...")
        with open(self.args.data_path, 'w') as f:
            for sample in supervised_samples:
                f.write(json.dumps(sample) + "\n")

        # Save DPO dataset
        print(f"Saving {len(dpo_samples)} DPO samples...")
        dpo_data_path = os.path.join(self.args.datasets_dir, "comparison_data.json")
        os.makedirs(self.args.datasets_dir, exist_ok=True)
        with open(dpo_data_path, 'w') as f:
            for sample in dpo_samples:
                f.write(json.dumps(sample) + "\n")

        print(f"\n{'='*100}")
        print("📊 FINAL DATASET SUMMARY")
        print(f"{'='*100}")
        print(f"✅ Supervised Learning Samples: {len(supervised_samples)}")
        print(f"🔄 DPO Training Samples: {len(dpo_samples)}")
        print(f"📈 Total Samples Processed: {len(supervised_samples) + len(dpo_samples)}")
        print(f"{'='*100}")

        # Train supervised policy
        if len(supervised_samples) > 0:
            print("Training supervised policy...")
            supervised_finetune(self.args)
            merge_peft_adapter(model_name=self.args.output_path, output_name=self.args.rl_base_model)
        else:
            print("No supervised samples - skipping supervised training")

        # Train DPO policy
        if len(dpo_samples) > 0:
            print("Training DPO policy...")
            tuning_lm_with_dpo(self.args)
            merge_peft_adapter(model_name=self.args.output_dir+"dpo_model", output_name="./saved_models/sep_model")
        else:
            print("No DPO samples - skipping DPO training")

    def _get_news_summary(self, date_str):
        """Get news summary for a specific date"""
        # TODO: Replace placeholder with actual retrieval; until then, return None to avoid synthetic data
        return None

    def test(self):
        # Load test data
        print("Loading Test Data...")
        data = self.dataloader.load(flag="test")
        
        if len(data) == 0:
            print("ERROR: No test data loaded!")
            return
        
        print(f"Testing on {len(data)} samples...")
        
        # Create test dual agent (uses fine-tuned model)
        test_dual_agent = DualPredictionAgent()
        test_dual_agent.model_manager.test_mode = True
        
        # Initialize evaluation metrics
        evaluator = EvaluationMetrics()
        
        # Test predictions using dual model architecture
        test_results = []
        
        for idx, row in data.iterrows():
            print(f"Testing sample {idx + 1}/{len(data)}")
            
            # Build tweets-based summary for test as well (not used in prediction when testing candle-only)
            # We still build it to keep input parity if needed later, but will pass an empty summary to news path
            btc_texts = self.dataloader.get_tweets_texts(row['date'], self.dataloader.main_market)
            gold_texts = []
            for m in self.dataloader.secondary_markets:
                gold_texts.extend(self.dataloader.get_tweets_texts(row['date'], m))
            all_texts = [t for t in (btc_texts + gold_texts) if t]
            if not all_texts:
                print("Skipping test sample: no tweets to summarize")
                continue
            # We won't use news summary in prediction in test phase per requirement
            
            # Get predictions from candle model only by calling predict and then ignoring news outputs
            result = test_dual_agent.predict(
                market_data=row['market_data'],
                sentiment_data={
                    'btc_sentiment': row['market_data']['btc_sentiment'],
                    'gold_sentiment': row['market_data']['gold_sentiment']
                },
                news_summary="",
                target=row['target']  # Add target parameter
            )
            
            # Overwrite final_prediction/explanation to candle-only for reporting
            final_pred = result['candle_prediction']
            final_exp = result['candle_explanation']
            
            # Check if prediction is correct (using target from data, not from model input)
            is_correct = final_pred.lower() == row['target'].lower()
            
            # Add to evaluator for comprehensive metrics
            evaluator.add_prediction(row['target'], final_pred)
            
            test_result = {
                'date': row['date'],
                'target': row['target'],
                'final_prediction': final_pred,
                'candle_prediction': result['candle_prediction'],
                'news_prediction': result['news_prediction'],
                'models_agree': result['models_agree'],
                'is_correct': is_correct,
                'final_explanation': final_exp
            }
            
            test_results.append(test_result)
            
            print(f"Sample {idx + 1}: Target={row['target']}, Prediction={final_pred}, Correct={is_correct}")

        # Calculate comprehensive metrics
        metrics = evaluator.calculate_metrics()
        
        print(f'\n=== COMPREHENSIVE TEST RESULTS ===')
        evaluator.print_report()
        
        # Save detailed results
        os.makedirs(self.args.save_dir, exist_ok=True)
        
        # Save test results
        test_results_path = os.path.join(self.args.save_dir, "test_results.json")
        with open(test_results_path, 'w') as f:
            for result in test_results:
                f.write(json.dumps(result) + "\n")
        
        # Save comprehensive metrics
        metrics_path = os.path.join(self.args.save_dir, "evaluation_metrics.json")
        evaluator.save_results(metrics_path)
        
        # Save summary
        summary_path = os.path.join(self.args.save_dir, "test_summary.json")
        summary = {
            'total_samples': len(test_results),
            'correct_predictions': metrics['true_positives'] + metrics['true_negatives'],
            'incorrect_predictions': metrics['false_positives'] + metrics['false_negatives'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'mcc': metrics['mcc'],
            'specificity': metrics['specificity'],
            'sensitivity': metrics['sensitivity'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'roc_auc': metrics['roc_auc'],
            'models_agreement_rate': sum(1 for r in test_results if r['models_agree']) / len(test_results) if len(test_results) > 0 else 0
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate plots
        confusion_matrix_path = os.path.join(self.args.save_dir, "confusion_matrix.png")
        evaluator.plot_confusion_matrix(confusion_matrix_path)
        
        roc_curve_path = os.path.join(self.args.save_dir, "roc_curve.png")
        evaluator.plot_roc_curve(roc_curve_path)
        
        pr_curve_path = os.path.join(self.args.save_dir, "precision_recall_curve.png")
        evaluator.plot_precision_recall_curve(pr_curve_path)
        
        print(f'\n=== FINAL TEST SUMMARY ===')
        print(f'Total samples processed: {len(test_results)}')
        print(f'Accuracy: {metrics["accuracy"]:.4f}')
        print(f'Precision: {metrics["precision"]:.4f}')
        print(f'Recall: {metrics["recall"]:.4f}')
        print(f'F1-Score: {metrics["f1_score"]:.4f}')
        print(f'MCC: {metrics["mcc"]:.4f}')
        print(f'Models agreement rate: {summary["models_agreement_rate"]:.4f}')

