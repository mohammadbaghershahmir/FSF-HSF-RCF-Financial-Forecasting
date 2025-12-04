from typing import Dict, List, Tuple, Optional
import random
from utils.llm import FastChatLLM
from utils.enhanced_prompts import REFLECTION_TYPES
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_loader import VicunaModel
import torch


class ReflectionAgent:
    """Agent that performs different types of reflections to generate diverse DPO training pairs"""
    
    def __init__(self, model_manager=None, strategy="balanced"):
        self.model_manager = model_manager
        self.llm = None
        self.strategy = strategy
        self.all_reflection_types = list(REFLECTION_TYPES.keys())
        
        # Define reflection strategies
        self.strategies = {
            "all": self.all_reflection_types,
            "balanced": ["retry", "keywords", "advice", "solution"],
            "simple": ["retry", "keywords"],
            "complex": ["advice", "explanation", "instructions", "solution", "composite"],
            "technical": ["retry", "keywords", "advice"],
            "sentiment": ["explanation", "instructions", "composite"]
        }
        
        self.reflection_types = self.strategies.get(strategy, self.strategies["balanced"])
        
    def load_model(self):
        """Load the reflection model"""
        if self.model_manager is None:
            print("Loading Reflection Model...")
            model_data = VicunaModel.get_model(
                model_name="lmsys/vicuna-7b-v1.5",
                quantization=True,
                device_map="auto"
            )
            self.model_manager = model_data
            print("Reflection Model Loaded.")
        
        # Handle both dict and object types
        if isinstance(self.model_manager, dict):
            self.llm = FastChatLLM(model=self.model_manager["model"], tokenizer=self.model_manager["tokenizer"])
        else:
            # If it's a DualModelManager object, load the candle model
            model_data = self.model_manager.load_candle_model()
            self.llm = FastChatLLM(model=model_data["model"], tokenizer=model_data["tokenizer"])
        
    def unload_model(self):
        """Unload model to free GPU memory"""
        if self.model_manager is not None:
            if isinstance(self.model_manager, dict):
                if hasattr(self.model_manager, 'model'):
                    del self.model_manager['model']
                if hasattr(self.model_manager, 'tokenizer'):
                    del self.model_manager['tokenizer']
            else:
                # If it's a DualModelManager object, unload current model
                self.model_manager.unload_current_model()
            torch.cuda.empty_cache()
            self.model_manager = None
            print("Reflection model unloaded from GPU memory.")
    
    def reflect(self, 
                original_prediction: str, 
                original_explanation: str, 
                correct_prediction: str, 
                correct_explanation: str,
                reflection_type: str = None,
                market_data: Dict = None,
                sentiment_data: Dict = None,
                news_summary: str = None) -> Dict:
        """
        Perform reflection using specified type or random type
        
        Args:
            original_prediction: The incorrect prediction
            original_explanation: The incorrect explanation
            correct_prediction: The correct prediction
            correct_explanation: The correct explanation
            reflection_type: Type of reflection to use (if None, random)
            market_data: Market data for context
            sentiment_data: Sentiment data for context
            news_summary: News summary for context
            
        Returns:
            Dict with reflection results
        """
        if reflection_type is None:
            reflection_type = random.choice(self.all_reflection_types)
        
        if reflection_type not in self.all_reflection_types:
            raise ValueError(f"Invalid reflection type: {reflection_type}")
        
        # Load model if not already loaded
        if self.llm is None:
            self.load_model()
        
        # Get the reflection prompt
        reflection_prompt = REFLECTION_TYPES[reflection_type]
        
        # Format the prompt with context
        formatted_prompt = self._format_reflection_prompt(
            reflection_prompt,
            original_prediction,
            original_explanation,
            correct_prediction,
            correct_explanation,
            market_data,
            sentiment_data,
            news_summary
        )
        
        print(f"\n{'='*80}")
        print(f"🔄 REFLECTION AGENT - TYPE: {reflection_type.upper()}")
        print(f"{'='*80}")
        print(f"📝 Original Prediction: {original_prediction}")
        print(f"✅ Correct Prediction: {correct_prediction}")
        print(f"🤖 Reflection Prompt Length: {len(formatted_prompt)} characters")
        
        # Generate reflection
        print(f"\n🔄 GENERATING REFLECTION...")
        reflection_response = self.llm(formatted_prompt)
        
        # Parse reflection response
        reflection_prediction, reflection_explanation = self._parse_reflection_response(reflection_response)
        
        print(f"\n📋 REFLECTION OUTPUT:")
        print(f"Prediction: {reflection_prediction}")
        print(f"Explanation Length: {len(reflection_explanation)} characters")
        print(f"{'='*80}")
        
        return {
            "reflection_type": reflection_type,
            "original_prediction": original_prediction,
            "original_explanation": original_explanation,
            "correct_prediction": correct_prediction,
            "correct_explanation": correct_explanation,
            "reflection_prediction": reflection_prediction,
            "reflection_explanation": reflection_explanation,
            "reflection_response": reflection_response
        }
    
    def select_reflection_types(self, original_prediction, correct_prediction, market_data, sentiment_data):
        """
        Select reflection types based on strategy and context
        
        Args:
            original_prediction: The incorrect prediction
            correct_prediction: The correct prediction
            market_data: Market data for context
            sentiment_data: Sentiment data for context
            
        Returns:
            List of selected reflection types
        """
        if self.strategy == "adaptive":
            # Adaptive selection based on context
            selected_types = ["retry"]  # Always include retry
            
            # Add technical types if market data is rich
            if market_data and len(str(market_data)) > 100:
                selected_types.extend(["keywords", "advice"])
            
            # Add sentiment types if sentiment data is rich
            if sentiment_data and len(str(sentiment_data)) > 50:
                selected_types.extend(["explanation", "instructions"])
            
            # Add complex types if prediction error is significant
            if original_prediction != correct_prediction:
                selected_types.extend(["solution"])
            
            # Limit to 4 types maximum
            return selected_types[:4]
        
        elif self.strategy == "progressive":
            # Progressive selection: start simple, add complex if needed
            return ["retry", "keywords", "advice", "solution"]
        
        elif self.strategy == "quality":
            # Quality-based selection: only high-success types
            return ["retry", "keywords", "advice", "solution"]
        
        else:
            # Use predefined strategy
            return self.reflection_types
    
    def reflect_with_selected_types(self, 
                                   original_prediction: str,
                                   original_explanation: str,
                                   correct_prediction: str,
                                   correct_explanation: str,
                                   market_data: Dict = None,
                                   sentiment_data: Dict = None,
                                   news_summary: str = None) -> List[Dict]:
        """
        Perform reflections using selected types based on strategy
        
        Args:
            original_prediction: The incorrect prediction
            original_explanation: The incorrect explanation
            correct_prediction: The correct prediction
            correct_explanation: The correct explanation
            market_data: Market data for context
            sentiment_data: Sentiment data for context
            news_summary: News summary for context
            
        Returns:
            List of reflection results
        """
        # Select reflection types based on strategy
        selected_types = self.select_reflection_types(
            original_prediction, correct_prediction, market_data, sentiment_data
        )
        
        print(f"\n{'='*100}")
        print(f"🔄 REFLECTION AGENT - STRATEGY: {self.strategy.upper()}")
        print(f"📋 Selected Types: {selected_types}")
        print(f"{'='*100}")
        
        reflection_results = []
        
        for reflection_type in selected_types:
            try:
                result = self.reflect(
                    original_prediction=original_prediction,
                    original_explanation=original_explanation,
                    correct_prediction=correct_prediction,
                    correct_explanation=correct_explanation,
                    reflection_type=reflection_type,
                    market_data=market_data,
                    sentiment_data=sentiment_data,
                    news_summary=news_summary
                )
                
                # Check if reflection is correct
                is_correct = result["reflection_prediction"] == correct_prediction
                result["is_correct"] = is_correct
                
                reflection_results.append(result)
                
                print(f"✅ {reflection_type.upper()}: {'CORRECT' if is_correct else 'INCORRECT'}")
                
            except Exception as e:
                print(f"❌ Error in {reflection_type} reflection: {e}")
                continue
        
        print(f"\n📊 REFLECTION SUMMARY:")
        print(f"�� Total Reflections: {len(reflection_results)}")
        print(f"📈 Success Rate: {sum(1 for r in reflection_results if r['is_correct'])/len(reflection_results)*100:.1f}%")
        print(f"{'='*100}")
        
        return reflection_results
    
    def _format_reflection_prompt(self, 
                                reflection_prompt: str,
                                original_prediction: str,
                                original_explanation: str,
                                correct_prediction: str,
                                correct_explanation: str,
                                market_data: Dict = None,
                                sentiment_data: Dict = None,
                                news_summary: str = None) -> str:
        """Format the reflection prompt with all necessary context"""
        
        # Base formatting
        formatted = reflection_prompt.format(
            original_prediction=original_prediction,
            original_explanation=original_explanation,
            correct_prediction=correct_prediction,
            correct_explanation=correct_explanation
        )
        
        # Add additional context if available
        if market_data or sentiment_data or news_summary:
            formatted += "\n\n=== ADDITIONAL CONTEXT ===\n"
            
            if market_data:
                formatted += f"Market Data: {market_data}\n"
            
            if sentiment_data:
                formatted += f"Sentiment Data: {sentiment_data}\n"
            
            if news_summary:
                formatted += f"News Summary: {news_summary}\n"
        
        return formatted
    
    def _parse_reflection_response(self, response: str) -> Tuple[str, str]:
        """Parse the reflection response to extract prediction and explanation - ONLY Positive or Negative allowed"""
        response = response.strip()
        
        # Look for exact matches first
        if response.lower() == "positive":
            return "Positive", response
        elif response.lower() == "negative":
            return "Negative", response
        
        # Look for the standard format: "Bitcoin Price Movement: [Positive/Negative]"
        lines = response.split('\n')
        prediction = "Unknown"
        explanation = response
        
        for line in lines:
            if "Bitcoin Price Movement:" in line:
                if "Positive" in line:
                    prediction = "Positive"
                elif "Negative" in line:
                    prediction = "Negative"
                break
        
        # If no prediction found, try alternative patterns
        if prediction == "Unknown":
            for line in lines:
                if "Price Movement:" in line:
                    if "Positive" in line:
                        prediction = "Positive"
                    elif "Negative" in line:
                        prediction = "Negative"
                    break
        
        # If still no prediction, check the entire response
        if prediction == "Unknown":
            if "positive" in response.lower():
                prediction = "Positive"
            elif "negative" in response.lower():
                prediction = "Negative"
        
        # Default to Positive if still unknown (to avoid crashes)
        if prediction == "Unknown":
            prediction = "Positive"
        
        return prediction, explanation
    
    def generate_dpo_pairs(self, 
                          incorrect_prediction: str,
                          incorrect_explanation: str,
                          correct_prediction: str,
                          correct_explanation: str,
                          num_reflections: int = 3,
                          market_data: Dict = None,
                          sentiment_data: Dict = None,
                          news_summary: str = None) -> List[Dict]:
        """
        Generate multiple DPO pairs using different reflection types
        
        Args:
            incorrect_prediction: The incorrect prediction
            incorrect_explanation: The incorrect explanation
            correct_prediction: The correct prediction
            correct_explanation: The correct explanation
            num_reflections: Number of different reflection types to use
            market_data: Market data for context
            sentiment_data: Sentiment data for context
            news_summary: News summary for context
            
        Returns:
            List of DPO pairs
        """
        dpo_pairs = []
        
        # Select random reflection types
        selected_types = random.sample(self.all_reflection_types, min(num_reflections, len(self.all_reflection_types)))
        
        for reflection_type in selected_types:
            try:
                reflection_result = self.reflect(
                    original_prediction=incorrect_prediction,
                    original_explanation=incorrect_explanation,
                    correct_prediction=correct_prediction,
                    correct_explanation=correct_explanation,
                    reflection_type=reflection_type,
                    market_data=market_data,
                    sentiment_data=sentiment_data,
                    news_summary=news_summary
                )
                
                # Create DPO pair
                dpo_pair = {
                    "prompt": "Analyze Bitcoin and Gold market data to predict Bitcoin's price movement.",
                    "chosen": correct_explanation,  # The correct explanation
                    "rejected": reflection_result["reflection_explanation"],  # The reflection explanation
                    "preferred": "chosen",
                    "reflection_type": reflection_type,
                    "original_prediction": incorrect_prediction,
                    "correct_prediction": correct_prediction
                }
                
                dpo_pairs.append(dpo_pair)
                
            except Exception as e:
                print(f"Error generating reflection for type {reflection_type}: {e}")
                continue
        
        return dpo_pairs
