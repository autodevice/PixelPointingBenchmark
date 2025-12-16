import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from evaluation.vlm_evaluator import VLMEvaluator
from evaluation.metrics import calculate_distance, calculate_accuracy_metrics, calculate_statistics_across_passes
from evaluation.results_manager import ResultsManager
from test_suites.registry import registry


MODELS = {
    "sonnet": "anthropic/claude-sonnet-4-20250514",
    "opus": "anthropic/claude-opus-4-5-20251101",
    "gemini3": "gemini/gemini-3-pro-preview",
    "chatgpt": "openai/gpt-5.1",
    "haiku": "anthropic/claude-haiku-4-5-20251001",
    "qwen3-vl": "openrouter/qwen/qwen3-vl-235b-a22b-instruct",
    "glm-4.6v": "openrouter/z-ai/glm-4.6v",
    "gemini-2.5-flash": "gemini/gemini-2.5-flash",
}


def run_evaluation(
    test_suite_name: str,
    models: List[str],
    output_dir: str,
    screen_width: int = 819,
    screen_height: int = 1456,
    num_passes: int = 1,
    save_images: bool = True,
) -> None:
    """Run evaluation for a test suite and models."""
    test_suite = registry.get(test_suite_name)
    if not test_suite:
        raise ValueError(f"Test suite '{test_suite_name}' not found. Available: {registry.list_all()}")
    
    results_manager = ResultsManager(output_dir)
    screen_size = {"name": "custom", "width": screen_width, "height": screen_height}
    
    suite_dir = Path(output_dir) / test_suite_name / screen_size["name"]
    images_dir = suite_dir / "images"
    if save_images:
        images_dir.mkdir(parents=True, exist_ok=True)
    
    test_configs = test_suite.get_test_cases()
    
    test_images = {}
    if save_images:
        for config in test_configs:
            image, expected_coords = test_suite.generate_test_image(config, screen_width, screen_height)
            img_path = images_dir / f"{config['name']}.png"
            Image.fromarray(image).save(img_path)
            test_images[config['name']] = (image, expected_coords)
    
    for model_name in models:
        if model_name not in MODELS:
            print(f"Warning: Unknown model {model_name}, skipping...")
            continue
        
        model_id = MODELS[model_name]
        evaluator = VLMEvaluator(model_id)
        
        all_passes_results = []
        
        for pass_num in range(1, num_passes + 1):
            print(f"\n{'='*60}")
            print(f"Evaluating {model_name} - Pass {pass_num}/{num_passes}")
            print(f"{'='*60}")
            
            model_results = []
            
            for i, config in enumerate(test_configs):
                print(f"\nTest {i+1}/{len(test_configs)}: {config['name']}")
                print(f"Prompt: {config['prompt']}")
                
                if config['name'] in test_images:
                    image, expected_coords = test_images[config['name']]
                else:
                    image, expected_coords = test_suite.generate_test_image(config, screen_width, screen_height)
                
                print("Querying model...")
                response, error, metadata = evaluator.query_model(image, config["prompt"])
                
                if metadata:
                    print(f"Image size: {metadata['size'][0]}x{metadata['size'][1]} "
                          f"(~{metadata['estimated_tokens']} tokens)")
                
                if error:
                    print(f"Error: {error}")
                    result = {
                        "test_name": config["name"],
                        "prompt": config["prompt"],
                        "expected_coords": expected_coords,
                        "response": None,
                        "predicted_coords": None,
                        "distance": None,
                        "error": str(error),
                        "image_metadata": metadata,
                    }
                else:
                    print(f"Response: {response}")
                    predicted_coords = evaluator.extract_coordinates(response)
                    
                    if predicted_coords:
                        distance = calculate_distance(predicted_coords, expected_coords)
                        print(f"Expected: {expected_coords}, Predicted: {predicted_coords}, Distance: {distance:.2f}px")
                        result = {
                            "test_name": config["name"],
                            "prompt": config["prompt"],
                            "expected_coords": expected_coords,
                            "response": response,
                            "predicted_coords": predicted_coords,
                            "distance": distance,
                            "error": None,
                            "image_metadata": metadata,
                        }
                    else:
                        print("Could not extract coordinates from response")
                        result = {
                            "test_name": config["name"],
                            "prompt": config["prompt"],
                            "expected_coords": expected_coords,
                            "response": response,
                            "predicted_coords": None,
                            "distance": None,
                            "error": "Could not extract coordinates",
                            "image_metadata": metadata,
                        }
                
                model_results.append(result)
            
            metrics = calculate_accuracy_metrics(model_results, (screen_width, screen_height))
            
            print(f"\n{'='*60}")
            print(f"Results for {model_name} - Pass {pass_num}:")
            print(f"{'='*60}")
            print(f"Total tests: {metrics['total_tests']}")
            print(f"Successful extractions: {metrics['successful_extractions']}")
            print(f"Extraction rate: {metrics['extraction_rate']:.2f}%")
            if metrics["mean_distance"] is not None:
                print(f"Mean distance: {metrics['mean_distance']:.2f}px")
                print(f"Median distance: {metrics['median_distance']:.2f}px")
                print(f"Max distance: {metrics['max_distance']:.2f}px")
                print(f"Min distance: {metrics['min_distance']:.2f}px")
                print(f"Accuracy within 10px: {metrics['accuracy_within_10px']:.2f}%")
                print(f"Accuracy within 5% screen: {metrics['accuracy_within_5_percent']:.2f}%")
            
            run_id = results_manager.save_run(
                test_suite_name,
                model_name,
                model_id,
                screen_size,
                test_configs,
                model_results,
                metrics,
                pass_number=pass_num if num_passes > 1 else None,
            )
            
            all_passes_results.append(model_results)
        
        if num_passes > 1:
            stats = calculate_statistics_across_passes(all_passes_results, (screen_width, screen_height))
            print(f"\n{'='*60}")
            print(f"Statistics across {num_passes} passes for {model_name}:")
            print(f"{'='*60}")
            if stats:
                print(f"Overall mean distance: {stats.get('overall_mean_distance', 0):.2f}px")
                print(f"Standard deviation: {stats.get('overall_std_distance', 0):.2f}px")
                print(f"Mean distance across passes: {stats.get('mean_distance_across_passes', 0):.2f}px")
                print(f"Std dev across passes: {stats.get('std_distance_across_passes', 0):.2f}px")
        
        results_manager.consolidate_results(test_suite_name, screen_size["name"], models)
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to: {output_dir}")
    print(f"{'='*60}")

