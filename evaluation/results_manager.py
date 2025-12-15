import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


class ResultsManager:
    """Manages evaluation results with support for multiple passes."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def get_run_id(self, test_suite: str, model: str, screen_size: str) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{test_suite}_{model}_{screen_size}_{timestamp}"
    
    def save_run(
        self,
        test_suite: str,
        model: str,
        model_id: str,
        screen_size: Dict[str, Any],
        test_configs: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        pass_number: Optional[int] = None,
    ) -> str:
        """Save a single run's results. Returns the run ID."""
        run_id = self.get_run_id(test_suite, model, screen_size["name"])
        
        suite_dir = self.results_dir / test_suite / screen_size["name"]
        suite_dir.mkdir(parents=True, exist_ok=True)
        
        run_data = {
            "run_id": run_id,
            "test_suite": test_suite,
            "model": model,
            "model_id": model_id,
            "screen_size": screen_size,
            "pass_number": pass_number,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "results": results,
            "test_configs": test_configs,
        }
        
        if pass_number is not None:
            filename = f"{model}_pass{pass_number}_{run_id}.json"
        else:
            filename = f"{model}_{run_id}.json"
        
        run_file = suite_dir / filename
        with open(run_file, "w") as f:
            json.dump(run_data, f, indent=2)
        
        self._update_index(test_suite, screen_size["name"], model, run_id, pass_number)
        
        return run_id
    
    def _update_index(
        self, test_suite: str, screen_size: str, model: str, run_id: str, pass_number: Optional[int]
    ):
        """Update the index of all runs."""
        suite_dir = self.results_dir / test_suite / screen_size
        index_file = suite_dir / "runs_index.json"
        
        if index_file.exists():
            with open(index_file, "r") as f:
                index = json.load(f)
        else:
            index = {}
        
        if model not in index:
            index[model] = []
        
        index[model].append({
            "run_id": run_id,
            "pass_number": pass_number,
            "timestamp": datetime.now().isoformat(),
        })
        
        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)
    
    def load_runs(
        self, test_suite: str, screen_size: str, model: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load all runs for a test suite and screen size, optionally filtered by model."""
        suite_dir = self.results_dir / test_suite / screen_size
        
        if not suite_dir.exists():
            return {}
        
        index_file = suite_dir / "runs_index.json"
        if not index_file.exists():
            return {}
        
        with open(index_file, "r") as f:
            index = json.load(f)
        
        runs = {}
        for model_name, run_list in index.items():
            if model and model_name != model:
                continue
            
            runs[model_name] = []
            for run_info in run_list:
                run_id = run_info["run_id"]
                pass_num = run_info.get("pass_number")
                
                if pass_num is not None:
                    pattern = f"{model_name}_pass{pass_num}_*.json"
                else:
                    pattern = f"{model_name}_*.json"
                
                run_files = list(suite_dir.glob(pattern))
                for run_file in run_files:
                    if run_id in run_file.name:
                        with open(run_file, "r") as f:
                            runs[model_name].append(json.load(f))
                        break
        
        return runs
    
    def consolidate_results(
        self, test_suite: str, screen_size: str, models: List[str]
    ) -> Dict[str, Any]:
        """Consolidate all runs into a single results file for the viewer."""
        suite_dir = self.results_dir / test_suite / screen_size
        
        all_runs = self.load_runs(test_suite, screen_size)
        
        all_models_set = set()
        
        consolidated = {
            "test_suite": test_suite,
            "screen_size": {"name": screen_size, "width": None, "height": None},
            "models": {},
            "tests": [],
        }
        
        test_names = set()
        for model_runs in all_runs.values():
            for run in model_runs:
                if consolidated["screen_size"]["width"] is None and run.get("screen_size"):
                    consolidated["screen_size"]["width"] = run["screen_size"].get("width")
                    consolidated["screen_size"]["height"] = run["screen_size"].get("height")
                for result in run.get("results", []):
                    test_names.add(result["test_name"])
        
        test_names = sorted(test_names)
        
        for test_name in test_names:
            test_data = {
                "test_name": test_name,
                "prompt": None,
                "expected_coords": None,
                "image_file": f"images/{test_name}.png",
                "models": {},
            }
            
            for model in models:
                if model not in all_runs:
                    continue
                
                model_runs = all_runs[model]
                model_predictions = []
                
                for run in model_runs:
                    for result in run.get("results", []):
                        if result["test_name"] == test_name:
                            if test_data["prompt"] is None:
                                test_data["prompt"] = result.get("prompt")
                            if test_data["expected_coords"] is None:
                                test_data["expected_coords"] = result.get("expected_coords")
                            
                            if result.get("predicted_coords"):
                                model_predictions.append({
                                    "predicted_coords": result["predicted_coords"],
                                    "distance": result.get("distance"),
                                    "response": result.get("response"),
                                    "error": result.get("error"),
                                    "pass_number": run.get("pass_number"),
                                    "run_id": run.get("run_id"),
                                })
                
                if model_predictions:
                    test_data["models"][model] = {
                        "predictions": model_predictions,
                        "num_passes": len(set(p.get("pass_number") for p in model_predictions if p.get("pass_number"))),
                    }
                    all_models_set.add(model)
            
            if test_data["prompt"]:
                consolidated["tests"].append(test_data)
        
        consolidated["models"] = {model: model for model in sorted(all_models_set)}
        
        consolidated_file = suite_dir / "consolidated_results.json"
        with open(consolidated_file, "w") as f:
            json.dump(consolidated, f, indent=2)
        
        self._update_test_suites_index()
        
        return consolidated
    
    def _update_test_suites_index(self):
        """Update the test_suites.json index file."""
        self.update_test_suites_index()
    
    def update_test_suites_index(self):
        """Update the test_suites.json index file. Public method for manual updates."""
        if not self.results_dir.exists():
            return
        
        test_suites = []
        for item in self.results_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                test_suites.append(item.name)
        
        test_suites.sort()
        index = {"test_suites": test_suites}
        
        index_file = self.results_dir / "test_suites.json"
        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)
        
        return test_suites
    
    def fix_consolidated_results(self, test_suite: str, screen_size: str = "custom"):
        """Fix consolidated_results.json to include models list if missing."""
        suite_dir = self.results_dir / test_suite / screen_size
        consolidated_file = suite_dir / "consolidated_results.json"
        
        if not consolidated_file.exists():
            raise FileNotFoundError(f"Consolidated results not found: {consolidated_file}")
        
        with open(consolidated_file, "r") as f:
            data = json.load(f)
        
        all_models = set()
        for test in data.get("tests", []):
            all_models.update(test.get("models", {}).keys())
        
        data["models"] = {model: model for model in sorted(all_models)}
        
        with open(consolidated_file, "w") as f:
            json.dump(data, f, indent=2)
        
        return sorted(all_models)

