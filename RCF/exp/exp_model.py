from data_load.dataloader import DataLoader
from explain_module.util import summarize_trial, remove_reflections, save_results
from explain_module.dual_agents import DualPredictionAgent, DualModelManager, CandleSentimentAgent
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
                # DualModelManager imported above
        self.model_manager = DualModelManager()
        self.dual_agent = DualPredictionAgent(self.model_manager)  # Training mode

    def train(self):
        # Collect demonstration data using dual model architecture with reflection
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
        supervised_samples = []  # For merge_sample.json - only correct initial predictions
        dpo_samples = []         # For comparison_data.json - reflection pairs
        
        print("Starting dual model predictions with reflection process...")
        
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
            
            # Get predictions from both models with reflection process
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
            
            # ========================================
            # SUPERVISED LEARNING DATASET (merge_sample.json)
            # ========================================
            # Only add to supervised dataset if models agree and are correct
            if result['dataset_type'] == 'supervised' and row.get('target'):
                supervised_sample = {
                    "instruction": "Analyze Bitcoin and Gold market data to predict Bitcoin's price movement.",
                    "input": json.dumps({
                        "date": row['date'],
                        "market_data": row['market_data'],
                        "news_summary": news_summary
                    }, ensure_ascii=False),
                    "output": f"Bitcoin Price Movement: {result['final_prediction']}\n\nExplanation: {result['final_explanation']}",
                    "prediction": result['final_prediction'],
                    "target": row['target'],
                    "models_agree": result['models_agree'],
                    "candle_correct": result['candle_correct'],
                    "news_correct": result['news_correct']
                }
                supervised_samples.append(supervised_sample)
                print(f"✅ Sample {idx + 1}: Added to SUPERVISED dataset (models agreed and correct)")
            
            # ========================================
            # DPO DATASET (comparison_data.json)
            # ========================================
            # Add DPO pairs from reflection process for incorrect predictions
            elif result['dataset_type'] == 'dpo' and row.get('target'):
                dpo_list = self.dual_agent.create_dpo_sample(result)
                for s in dpo_list:
                    if all(k in s for k in ("prompt","chosen","rejected")) and s["chosen"] != s["rejected"]:
                        # Add metadata to DPO sample
                        s["target"] = row['target']
                        s["date"] = row['date']
                        s["original_candle_prediction"] = result['candle_prediction']
                        s["original_news_prediction"] = result['news_prediction']
                        s["models_agree"] = result['models_agree']
                        s["final_prediction"] = result['final_prediction']
                        dpo_samples.append(s)
                print(f"🔄 Sample {idx + 1}: Added {len(dpo_list)} DPO samples with ALL reflection types")
            
            else:
                print(f"⚠️  Sample {idx + 1}: Unknown dataset type - skipping")

        # ========================================
        # SAVE DATASETS
        # ========================================
        
        # Save supervised learning dataset (merge_sample.json)
        print(f"\n{'='*100}")
        print(f"💾 SAVING SUPERVISED LEARNING DATASET (merge_sample.json)")
        print(f"{'='*100}")
        print(f"📊 Total supervised samples: {len(supervised_samples)}")
        
        with open(self.args.data_path, 'w') as f:
            for sample in supervised_samples:
                f.write(json.dumps(sample) + "\n")
        
        # Save DPO dataset (comparison_data.json)
        print(f"\n{'='*100}")
        print(f"💾 SAVING DPO DATASET (comparison_data.json)")
        print(f"{'='*100}")
        print(f"📊 Total DPO samples: {len(dpo_samples)}")
        
        dpo_data_path = os.path.join(self.args.datasets_dir, "comparison_data.json")
        os.makedirs(self.args.datasets_dir, exist_ok=True)
        with open(dpo_data_path, 'w') as f:
            for sample in dpo_samples:
                f.write(json.dumps(sample) + "\n")

        # ========================================
        # DATASET ANALYSIS
        # ========================================
        print(f"\n{'='*100}")
        print("📊 FINAL DATASET ANALYSIS")
        print(f"{'='*100}")
        
        # Analyze supervised dataset
        supervised_correct = sum(1 for s in supervised_samples if s['prediction'].lower() == s['target'].lower())
        print(f"✅ Supervised Learning Dataset (merge_sample.json):")
        print(f"   - Total samples: {len(supervised_samples)}")
        print(f"   - Correct predictions: {supervised_correct}")
        print(f"   - Accuracy: {supervised_correct/len(supervised_samples)*100:.1f}%" if supervised_samples else "   - Accuracy: N/A")
        
        # Analyze DPO dataset
        print(f"\n🔄 DPO Training Dataset (comparison_data.json):")
        print(f"   - Total samples: {len(dpo_samples)}")
        
        # Analyze reflection types used
        reflection_types_used = {}
        correct_reflections = 0
        for sample in dpo_samples:
            if 'reflection_type' in sample:
                ref_type = sample['reflection_type']
                reflection_types_used[ref_type] = reflection_types_used.get(ref_type, 0) + 1
            if 'is_correct' in sample and sample['is_correct']:
                correct_reflections += 1
        
        if reflection_types_used:
            print(f"   - Reflection types used:")
            for ref_type, count in reflection_types_used.items():
                print(f"     * {ref_type}: {count} samples")
        
        print(f"   - Correct reflections: {correct_reflections}")
        print(f"   - Reflection success rate: {correct_reflections/len(dpo_samples)*100:.1f}%" if dpo_samples else "   - Reflection success rate: N/A")
        
        print(f"\n📈 Total Samples Processed: {len(supervised_samples) + len(dpo_samples)}")
        print(f"{'='*100}")

        # ========================================
        # TRAINING
        # ========================================
        
        # Train supervised policy
        if len(supervised_samples) > 0:
            print("\n�� Training supervised policy...")
            supervised_finetune(self.args)
            merge_peft_adapter(model_name=self.args.output_path, output_name=self.args.rl_base_model)
            print("✅ Supervised training completed")
        else:
            print("⚠️  No supervised samples - skipping supervised training")

        # Train DPO policy with reflection-enhanced data
        if len(dpo_samples) > 0:
            print("\n🚀 Training DPO policy with reflection-enhanced data...")
            tuning_lm_with_dpo(self.args)
            merge_peft_adapter(model_name=self.args.output_dir+"dpo_model", output_name="./saved_models/sep_model")
            print("✅ DPO training completed")
        else:
            print("⚠️  No DPO samples - skipping DPO training")

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
        
        # ========================================
        # LOAD DPO-TRAINED MODEL FOR TESTING
        # ========================================
        print("\n🚀 Loading DPO-trained model for testing...")
        
        # Check if DPO model exists
        dpo_model_path = "./saved_models/lora-Tiny-Vicuna"
        if not os.path.exists(dpo_model_path):
            print(f"❌ DPO model not found at {dpo_model_path}")
            print("Please run training first to generate the DPO model.")
            return
        
        # Load DPO-trained model
        try:
            from model_loader import VicunaModel
            dpo_model_data = VicunaModel.get_model(
                model_name=dpo_model_path,
                quantization=True,
                device_map="auto"
            )
            print("✅ DPO-trained model (lora-Tiny-Vicuna) loaded successfully")
        except Exception as e:
            print(f"❌ Error loading DPO model: {e}")
            print("Falling back to original model...")
            dpo_model_data = None
        
        # ========================================
        # CREATE SINGLE CANDLE MODEL FOR TESTING
        # ========================================
        print("\n🕯️  Creating Candle + Sentiment Model for testing...")
        
        # Create a single model manager for candle model only
                # DualModelManager imported above, CandleSentimentAgent
        
        test_model_manager = DualModelManager()
        test_model_manager.test_mode = True
        
        # Replace the candle model with DPO-trained model if available
        if dpo_model_data:
            test_model_manager.candle_model = dpo_model_data
            print("✅ Using DPO-trained model (lora-Tiny-Vicuna) for Candle + Sentiment testing")
        else:
            print("⚠️  Using original model for Candle + Sentiment testing")
        
        # Create only the candle agent
        test_candle_agent = CandleSentimentAgent(test_model_manager)
        
        # Initialize evaluation metrics
        evaluator = EvaluationMetrics()
        
        # Test predictions using ONLY Candle + Sentiment model
        test_results = []
        
        for idx, row in data.iterrows():
            print(f"\n{'='*80}")
            print(f"🧪 TESTING SAMPLE {idx + 1}/{len(data)} - DATE: {row['date']}")
            print(f"{'='*80}")
            
            # Get prediction from ONLY Candle + Sentiment model
            candle_prediction, candle_explanation = test_candle_agent.predict(
                market_data=row['market_data'],
                sentiment_data={
                    'btc_sentiment': row['market_data']['btc_sentiment'],
                    'gold_sentiment': row['market_data']['gold_sentiment']
                }
            )
            
            # Use candle prediction as final prediction
            final_pred = candle_prediction
            final_exp = candle_explanation
            
            # Check if prediction is correct
            is_correct = final_pred.lower() == row['target'].lower()
            
            # Add to evaluator for comprehensive metrics
            evaluator.add_prediction(row['target'], final_pred)
            
            test_result = {
                'date': row['date'],
                'target': row['target'],
                'final_prediction': final_pred,
                'candle_prediction': candle_prediction,
                'is_correct': is_correct,
                'final_explanation': final_exp,
                'model_type': 'DPO-trained (lora-Tiny-Vicuna)' if dpo_model_data else 'Original',
                'model_used': 'Candle + Sentiment Only'
            }
            
            test_results.append(test_result)
            
            print(f"🎯 Target: {row['target']}")
            print(f"🤖 Prediction: {final_pred}")
            print(f"✅ Correct: {is_correct}")
            print(f"📄 Explanation: {final_exp}")
            print(f"🕯️  Model Used: Candle + Sentiment (DPO-trained)")

        # Calculate comprehensive metrics
        metrics = evaluator.calculate_metrics()
        
        print("\n" + "="*100)
        print("📊 COMPREHENSIVE TEST RESULTS")
        print("="*100)
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
            'model_type': 'DPO-trained (lora-Tiny-Vicuna)' if dpo_model_data else 'Original',
            'model_used': 'Candle + Sentiment Only'
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
        
        print("\n" + "="*100)
        print("🏁 FINAL TEST SUMMARY")
        print("="*100)
        print(f'📊 Model Type: {summary["model_type"]}')
        print(f'🕯️  Model Used: {summary["model_used"]}')
        print(f'📈 Total samples processed: {len(test_results)}')
        print(f'🎯 Accuracy: {metrics["accuracy"]:.4f}')
        print(f'📊 Precision: {metrics["precision"]:.4f}')
        print(f'📊 Recall: {metrics["recall"]:.4f}')
        print(f'📊 F1-Score: {metrics["f1_score"]:.4f}')
        print(f'📊 MCC: {metrics["mcc"]:.4f}')
        print("="*100)
