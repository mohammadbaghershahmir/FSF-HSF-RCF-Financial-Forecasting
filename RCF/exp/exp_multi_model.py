from data_load.dataloader import DataLoader
from explain_module.util import summarize_trial, remove_reflections, save_results
from explain_module.multi_model_agents import MultiModelPredictionAgent
from predict_module.merge_peft_adapter import merge_peft_adapter
from predict_module.supervised_finetune import supervised_finetune
from predict_module.tuning_lm_with_rl import tuning_lm_with_dpo
from transformers import LlamaTokenizer, pipeline
from trl import AutoModelForCausalLMWithValueHead
from utils.evaluation_metrics import EvaluationMetrics
import os, json
import pandas as pd


class Exp_MultiModel:
    def __init__(self, args):
        self.args = args
        self.dataloader = DataLoader(args)
        # MultiModelPredictionAgent with configurable number of models
        self.num_models = getattr(args, 'num_models', 3)  # Default to 3 models
        if self.num_models % 2 == 0:
            self.num_models = 3  # Ensure odd number
            print(f"⚠️  Number of models must be odd. Setting to {self.num_models}")
        
        self.multi_agent = MultiModelPredictionAgent(self.num_models)

    def train(self):
        """Collect demonstration data using multi-model architecture with voting and reflection"""
        
        # Load data
        data = self.dataloader.load(flag="train")
        print(f"📊 Loaded {len(data)} samples for training")
        
        # Initialize datasets
        supervised_data = []
        dpo_data = []
        
        print("Starting multi-model predictions with voting and reflection process...")
        
        for i, (_, row) in enumerate(data.iterrows(), start=1):
            print(f"\n{'='*100}")
            print(f"📅 PROCESSING SAMPLE {i + 1}/{len(data)} - DATE: {row['date']}")
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
                try:
                    raw_summary = self.dataloader.summarizer.get_summary(self.dataloader.main_market, all_texts)
                    if raw_summary and self.dataloader.summarizer.is_informative(raw_summary):
                        news_summary = raw_summary
                    else:
                        print("Summarization non-informative → using neutral message for this day")
                        news_summary = "No meaningful market news found for the day."
                except Exception as e:
                    print(f"Summarization error: {e} → using neutral message (no synthetic summary)")
                    news_summary = "No meaningful market news found for the day."
            else:
                print("No tweets available → using neutral message (no news summary)")
                news_summary = "No meaningful market news found for the day."
            
            print("\n[DEBUG] Raw market_data keys:", list(row['market_data'].keys()) if isinstance(row.get('market_data'), dict) else type(row.get('market_data')))
            print("[DEBUG] Raw market_data sample:", row.get('market_data'))
            # Get predictions from multi-model ensemble with voting
            result = self.multi_agent.predict(
                market_data=row['market_data'],
                sentiment_data={
                    'btc_sentiment': row['market_data'].get('btc_sentiment', {}),
                    'gold_sentiment': row['market_data'].get('gold_sentiment', {})
                },
                news_summary=news_summary,
                target=row.get('target') or None
            )
            if result.get('dataset_type') == 'skip':
                print("Skipping: no valid model outputs (all missing explanation)")
                continue
            
            # Require valid prediction and explanation; skip otherwise
            if (result.get('dataset_type') == 'skip') or (not result.get('final_prediction') or result['final_prediction'] not in ("Positive", "Negative")):
                print("Skipping: invalid model results (missing prediction/explanation)")
                continue
            
            # ========================================
            # SUPERVISED LEARNING DATASET (merge_sample.json)
            # ========================================
            # Only add to supervised dataset if majority is correct
            if result['dataset_type'] == 'supervised':
                print(f'✅ Adding to supervised dataset: Majority correct')
                # Add each correct model's prediction and explanation as separate entries
                for idx_correct in result.get('correct_model_indices', []):
                    supervised_entry = {
                        'instruction': "Analyze Bitcoin and Gold market data to predict Bitcoin's price movement.",
                        'input': f"Market Data: {row['market_data']}\nSentiment: {result.get('sentiment_data', {})}\nNews: {news_summary}",
                        'output': f"Bitcoin Price Movement: {result['model_predictions'][idx_correct]}\n\nExplanation: {result['model_explanations'][idx_correct]}",
                        'date': row['date'],
                        'target': row.get('target'),
                        'model_index': int(idx_correct),
                        'model_predictions': result['model_predictions'],
                        'vote_counts': result['vote_counts'],
                        'majority_count': result['majority_count']
                    }
                    supervised_data.append(supervised_entry)
            
            # ========================================
            # DPO DATASET (comparison_data.json)
            # ========================================
            # Add DPO pairs if majority is wrong and reflections were generated
            elif result['dataset_type'] == 'dpo' and result.get('dpo_pairs'):
                print(f"🔄 Adding to DPO dataset: Majority wrong, {len(result['dpo_pairs'])} pairs generated")
                for dpo_pair in result['dpo_pairs']:
                    dpo_entry = {
                        **dpo_pair,
                        "date": row['date'],
                        "target": row.get('target'),
                        "market_data": row['market_data'],
                        "sentiment_data": result.get('sentiment_data', {}),
                        "news_summary": news_summary,
                        "model_predictions": result['model_predictions'],
                        "vote_counts": result['vote_counts'],
                        "majority_prediction": result['majority_prediction']
                    }
                    dpo_data.append(dpo_entry)
            
            print(f"📊 Current datasets: Supervised={len(supervised_data)}, DPO={len(dpo_data)}")
        
        # Save datasets
        self._save_datasets(supervised_data, dpo_data)
        
        print(f"\n{'='*100}")
        print("📊 TRAINING DATA COLLECTION SUMMARY")
        print(f"{'='*100}")
        print(f"✅ Supervised samples: {len(supervised_data)}")
        print(f"🔄 DPO pairs: {len(dpo_data)}")
        print(f"🤖 Models used: {self.num_models}")
        print(f"{'='*100}")
        
        # Optional: Run SFT and DPO training to display progress in terminal
        try:
            print(f"\n{'='*100}")
            print("🚀 STARTING SUPERVISED FINE-TUNING (SFT)")
            print(f"{'='*100}")
            supervised_finetune(self.args)
            print("✅ SFT COMPLETED")
        except Exception as e:
            print(f"⚠️ Skipping SFT (reason: {e})")
        
        try:
            print(f"\n{'='*100}")
            print("🤝 STARTING DPO TRAINING")
            print(f"Using dataset: {os.path.join(self.args.datasets_dir, 'comparison_data_multi_model.json')}")
            print(f"Base model: {self.args.rl_base_model}")
            print(f"Output dir: {self.args.output_dir}")
            print(f"{'='*100}")
            tuning_lm_with_dpo(self.args)
            print("✅ DPO TRAINING COMPLETED")
        except Exception as e:
            print(f"⚠️ Skipping DPO training (reason: {e})")
        
        return supervised_data, dpo_data

    def test(self):
        # Pre-test: unload all models and clear GPU cache to free memory for DPO model
        try:
            self.multi_agent.model_manager.unload_all_models()
        except Exception:
            pass
        import gc, torch
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        """Test the multi-model architecture"""
        print(f"\n{'='*100}")
        print(f"🧪 TESTING MULTI-MODEL ARCHITECTURE Ensemble MODELS)")
        print(f"{'='*100}")
        
        # Force single-agent DPO-only testing
        self.multi_agent = MultiModelPredictionAgent(num_models=1)
        self.multi_agent.model_manager.test_mode = True
        print("🧪 Test configured: single agent using DPO model only")
        
        # Load test data
        data = self.dataloader.load(flag="test")
        total = len(data)
        print(f"📊 Loaded {total} samples for testing")
        
        # Initialize evaluation metrics
        eval_metrics = EvaluationMetrics()
        
        test_results = []
        
        for i, (_, row) in enumerate(data.iterrows(), start=1):
            print(f"\n{'='*100}")
            print(f"📅 TESTING SAMPLE {i}/{total} - DATE: {row['date']}")
            print(f"{'='*100}")
            
            # Skip if essential fields missing
            if not row.get('market_data'):
                print("Missing market_data → passing empty structures to model")
                row['market_data'] = {}
            
            # Build tweets-based summary
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
                    print("Summarization non-informative → using neutral message for this day")
                    news_summary = "No meaningful market news found for the day."
            else:
                print("No tweets available → using neutral message (no news summary)")
                news_summary = "No meaningful market news found for the day."
            
            # Get prediction from multi-model ensemble
            result = self.multi_agent.predict(
                market_data=row['market_data'],
                sentiment_data={
                    'btc_sentiment': row['market_data'].get('btc_sentiment', {}),
                    'gold_sentiment': row['market_data'].get('gold_sentiment', {})
                },
                news_summary=news_summary,
                target=row.get('target') or None
            )
            if result.get('dataset_type') == 'skip':
                print("Skipping: no valid model outputs (all missing explanation)")
                continue
            
            # Record test result
            test_result = {
                "date": row['date'],
                "target": row.get('target'),
                "prediction": result['final_prediction'],
                "explanation": result['final_explanation'],
                "model_predictions": result['model_predictions'],
                "vote_counts": result['vote_counts'],
                "majority_count": result['majority_count'],
                "correct": result['final_prediction'] == row.get('target'),
                "correct_predictions": result['correct_predictions'],
                "incorrect_predictions": result['incorrect_predictions']
            }
            
            test_results.append(test_result)
            
            # Update evaluation metrics
            eval_metrics.add_prediction(row.get('target') or '', result['final_prediction'])
            
            print(f"🎯 Target: {row.get('target')}")
            print(f"🤖 Prediction: {result['final_prediction']}")
            print(f"✅ Correct: {test_result['correct']}")
            print(f"📊 Individual Models: {result['model_predictions']}")
            print(f"🗳️  Vote Counts: {result['vote_counts']}")
        
        # Calculate final metrics
        final_metrics = eval_metrics.calculate_metrics()
        
        print(f"\n{'='*100}")
        print("📊 TEST RESULTS SUMMARY")
        print(f"{'='*100}")
        print(f"🎯 Accuracy: {final_metrics['accuracy']:.3f}")
        print(f"📈 Precision: {final_metrics['precision']:.3f}")
        print(f"📉 Recall: {final_metrics['recall']:.3f}")
        print(f"🔄 F1-Score: {final_metrics['f1_score']:.3f}")
        print(f"🧮 MCC: {final_metrics['mcc']:.3f}")
        print(f"🤖 Models used: {self.num_models}")
        print(f"📊 Total samples: {len(test_results)}")
        print(f"{'='*100}")
        
        # Save test results
        self._save_test_results(test_results, final_metrics)
        
        return test_results, final_metrics

    def _save_datasets(self, supervised_data, dpo_data):
        """Save the collected datasets"""
        # Save supervised dataset
        if supervised_data:
            supervised_path = os.path.join(self.args.datasets_dir, "merge_sample_multi_model.json")
            with open(supervised_path, 'w', encoding='utf-8') as f:
                json.dump(supervised_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved supervised dataset: {supervised_path}")
        
        # Save DPO dataset
        if dpo_data:
            dpo_path = os.path.join(self.args.datasets_dir, "comparison_data_multi_model.json")
            with open(dpo_path, 'w', encoding='utf-8') as f:
                json.dump(dpo_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved DPO dataset: {dpo_path}")

    def _save_test_results(self, test_results, final_metrics):
        """Save test results"""
        results_path = os.path.join(self.args.save_dir, f"multi_model_test_results_{self.num_models}models.json")
        
        results_data = {
            "metrics": final_metrics,
            "num_models": self.num_models,
            "test_results": test_results
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved test results: {results_path}")
        
        # Also save as CSV for easy analysis
        csv_path = os.path.join(self.args.save_dir, f"multi_model_test_results_{self.num_models}models.csv")
        df = pd.DataFrame(test_results)
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved test results CSV: {csv_path}")
