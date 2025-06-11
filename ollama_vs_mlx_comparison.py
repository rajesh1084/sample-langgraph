import time
import statistics
import ollama
from mlx_lm import load, generate
from mlx_lm import sample_utils
import psutil
import gc
import subprocess
import sys
from typing import List, Dict, Any


class FairInferenceBenchmark:
    def __init__(self):
        self.ollama_model = "llama3.2:latest"
        self.mlx_model = (
            "mlx-community/Qwen2.5-3B-4bit"  # Using smaller model for fairness
        )
        self.mlx_model_obj = None
        self.mlx_tokenizer = None

    def check_ollama_running(self) -> bool:
        """Check if Ollama service is running"""
        try:
            result = subprocess.run(["pgrep", "ollama"], capture_output=True, text=True)
            return len(result.stdout.strip()) > 0
        except:
            return False

    def stop_ollama(self):
        """Stop Ollama service to free resources"""
        try:
            if self.check_ollama_running():
                print("Stopping Ollama service...")
                subprocess.run(["pkill", "ollama"], check=False)
                time.sleep(3)  # Wait for cleanup
                return True
        except Exception as e:
            print(f"Error stopping Ollama: {e}")
        return False

    def start_ollama(self):
        """Start Ollama service"""
        try:
            if not self.check_ollama_running():
                print("Starting Ollama service...")
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(5)  # Wait for startup
                print("  Ollama service started, checking model availability...")
                return self.check_ollama_model()
            else:
                print("Ollama service already running, checking model...")
                return self.check_ollama_model()
        except Exception as e:
            print(f"Error starting Ollama: {e}")
        return False

    def ensure_ollama_model(self):
        """Ensure Ollama model is available, pull if necessary"""
        if not self.check_ollama_model():
            print(f"  Model {self.ollama_model} not found, attempting to pull...")
            try:
                # Try to pull the model
                result = subprocess.run(
                    ["ollama", "pull", self.ollama_model],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )
                if result.returncode == 0:
                    print(f"  ✅ Successfully pulled {self.ollama_model}")
                    return self.check_ollama_model()
                else:
                    print(f"  ❌ Failed to pull model: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                print(f"  ⏰ Timeout pulling model {self.ollama_model}")
                return False
            except Exception as e:
                print(f"  ❌ Error pulling model: {e}")
                return False
        return True

    def unload_mlx_model(self):
        """Unload MLX model to free memory"""
        if self.mlx_model_obj is not None:
            print("Unloading MLX model...")
            self.mlx_model_obj = None
            self.mlx_tokenizer = None
            gc.collect()
            time.sleep(2)  # Allow cleanup

    def load_mlx_model(self):
        """Load MLX model once"""
        if self.mlx_model_obj is None:
            print("Loading MLX model...")
            start_time = time.time()
            try:
                self.mlx_model_obj, self.mlx_tokenizer = load(self.mlx_model)
                load_time = time.time() - start_time
                print(f"MLX model loaded in {load_time:.2f} seconds")
                return load_time
            except Exception as e:
                print(f"Failed to load MLX model: {e}")
                raise e
        return 0

    def check_ollama_model(self) -> bool:
        """Check if Ollama model is available"""
        try:
            print(f"  Checking Ollama model: {self.ollama_model}")
            response = ollama.generate(
                model=self.ollama_model, prompt="test", options={"num_predict": 1}
            )
            print(f"  ✅ Ollama model check successful")
            return True
        except Exception as e:
            print(f"  ❌ Ollama model not available: {e}")
            return False

    def get_system_resources(self) -> Dict[str, float]:
        """Get current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        return {
            "cpu_percent": cpu_percent,
            "memory_used_gb": memory.used / 1024**3,
            "memory_available_gb": memory.available / 1024**3,
            "memory_percent": memory.percent,
        }

    def ollama_inference(self, prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
        """Run Ollama inference and measure performance"""
        start_time = time.time()
        resources_before = self.get_system_resources()

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.2,
                    "num_predict": max_tokens,
                },
            )

            end_time = time.time()
            resources_after = self.get_system_resources()

            response_text = response["message"]["content"]
            tokens_generated = len(response_text.split())

            return {
                "response": response_text,
                "time": end_time - start_time,
                "tokens": tokens_generated,
                "tokens_per_second": (
                    tokens_generated / (end_time - start_time)
                    if (end_time - start_time) > 0
                    else 0
                ),
                "resources_before": resources_before,
                "resources_after": resources_after,
                "success": True,
            }

        except Exception as e:
            print(f"    ❌ Ollama inference error: {e}")
            return {
                "response": f"Error: {e}",
                "time": time.time() - start_time,
                "tokens": 0,
                "tokens_per_second": 0,
                "success": False,
            }

    def mlx_inference(self, prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
        """Run MLX inference and measure performance"""
        start_time = time.time()
        resources_before = self.get_system_resources()

        try:
            sampler = sample_utils.make_sampler(temp=0.2)
            response = generate(
                model=self.mlx_model_obj,
                tokenizer=self.mlx_tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
                sampler=sampler,
            )

            end_time = time.time()
            resources_after = self.get_system_resources()

            tokens_generated = len(response.split())

            return {
                "response": response,
                "time": end_time - start_time,
                "tokens": tokens_generated,
                "tokens_per_second": (
                    tokens_generated / (end_time - start_time)
                    if (end_time - start_time) > 0
                    else 0
                ),
                "resources_before": resources_before,
                "resources_after": resources_after,
                "success": True,
            }

        except Exception as e:
            print(f"    ❌ MLX inference error: {e}")
            return {
                "response": f"Error: {e}",
                "time": time.time() - start_time,
                "tokens": 0,
                "tokens_per_second": 0,
                "success": False,
            }

    def run_isolated_benchmark(
        self, prompts: List[str], iterations: int = 3
    ) -> Dict[str, Any]:
        """Run benchmark with proper resource isolation"""
        results = {
            "ollama": {
                "times": [],
                "tokens_per_sec": [],
                "responses": [],
                "resources": [],
            },
            "mlx": {
                "times": [],
                "tokens_per_sec": [],
                "responses": [],
                "resources": [],
            },
        }

        print(
            f"\nRunning ISOLATED benchmark with {len(prompts)} prompts, {iterations} iterations each"
        )
        print("=" * 70)

        # Phase 1: Ollama-only testing
        print("\n🔥 PHASE 1: OLLAMA TESTING (MLX models unloaded)")
        print("-" * 50)

        self.unload_mlx_model()  # Ensure MLX is unloaded

        # Better Ollama setup
        print("Setting up Ollama...")
        if self.start_ollama():
            print("Ensuring model is available...")
            ollama_available = self.ensure_ollama_model()
        else:
            ollama_available = False

        if ollama_available:
            print("✅ Ollama ready for testing")
            for i, prompt in enumerate(prompts):
                print(f"\nOllama Prompt {i+1}: {prompt[:50]}...")
                for iteration in range(iterations):
                    result = self.ollama_inference(prompt)
                    print(
                        f"  Iteration {iteration+1}: Success={result['success']}, Time={result['time']:.2f}s, TPS={result.get('tokens_per_second', 0):.1f}"
                    )
                    if result["success"]:
                        results["ollama"]["times"].append(result["time"])
                        results["ollama"]["tokens_per_sec"].append(
                            result["tokens_per_second"]
                        )
                        results["ollama"]["resources"].append(
                            {
                                "before": result["resources_before"],
                                "after": result["resources_after"],
                            }
                        )
                        if iteration == 0:
                            results["ollama"]["responses"].append(
                                result["response"][:100]
                            )
                    gc.collect()
        else:
            print("❌ Ollama not available for testing")

        # Phase 2: MLX-only testing
        print("\n🤖 PHASE 2: MLX TESTING (Ollama service stopped)")
        print("-" * 50)

        self.stop_ollama()  # Stop Ollama to free resources
        time.sleep(3)  # Allow system to settle

        try:
            mlx_load_time = self.load_mlx_model()
            mlx_available = True
            print("✅ MLX ready for testing")
        except Exception as e:
            print(f"❌ Failed to load MLX model: {e}")
            mlx_load_time = 0
            mlx_available = False

        if mlx_available:
            for i, prompt in enumerate(prompts):
                print(f"\nMLX Prompt {i+1}: {prompt[:50]}...")
                for iteration in range(iterations):
                    result = self.mlx_inference(prompt)
                    print(
                        f"  Iteration {iteration+1}: Success={result['success']}, Time={result['time']:.2f}s, TPS={result.get('tokens_per_second', 0):.1f}"
                    )
                    if result["success"]:
                        results["mlx"]["times"].append(result["time"])
                        results["mlx"]["tokens_per_sec"].append(
                            result["tokens_per_second"]
                        )
                        results["mlx"]["resources"].append(
                            {
                                "before": result["resources_before"],
                                "after": result["resources_after"],
                            }
                        )
                        if iteration == 0:
                            results["mlx"]["responses"].append(result["response"][:100])
                    gc.collect()

        return results, mlx_load_time

    def analyze_fair_results(self, results: Dict[str, Any], mlx_load_time: float):
        """Analyze results from isolated benchmark"""
        print("\n" + "=" * 60)
        print("FAIR BENCHMARK RESULTS (ISOLATED TESTING)")
        print("=" * 60)

        for engine in ["ollama", "mlx"]:
            if results[engine]["times"]:
                times = results[engine]["times"]
                tokens_per_sec = results[engine]["tokens_per_sec"]

                print(f"\n🔥 {engine.upper()} Performance (Isolated):")
                print(
                    f"  Average Time: {statistics.mean(times):.2f}s ± {statistics.stdev(times) if len(times) > 1 else 0:.2f}s"
                )
                print(f"  Median Time: {statistics.median(times):.2f}s")
                print(f"  Average Tokens/sec: {statistics.mean(tokens_per_sec):.2f}")
                print(f"  Min/Max Time: {min(times):.2f}s / {max(times):.2f}s")
                print(f"  Total successful runs: {len(times)}")

                # Resource usage analysis
                if results[engine]["resources"]:
                    cpu_usage = [
                        r["after"]["cpu_percent"] - r["before"]["cpu_percent"]
                        for r in results[engine]["resources"]
                    ]
                    memory_usage = [
                        r["after"]["memory_used_gb"] - r["before"]["memory_used_gb"]
                        for r in results[engine]["resources"]
                    ]
                    print(f"  Average CPU increase: {statistics.mean(cpu_usage):.1f}%")
                    print(
                        f"  Average Memory increase: {statistics.mean(memory_usage):.2f}GB"
                    )
            else:
                print(f"\n{engine.upper()} Performance:")
                print("  No successful runs recorded")

        # Model loading comparison
        print(f"\n📊 Model Loading:")
        print(f"  MLX Load Time: {mlx_load_time:.2f}s")
        print(f"  Ollama: Pre-loaded service (~0s)")

        # Debug information
        print(f"\nDebug Information:")
        print(f"  Ollama results count: {len(results['ollama']['times'])}")
        print(f"  MLX results count: {len(results['mlx']['times'])}")

        # Fair comparison
        print("\n🏆 FAIR COMPARISON:")
        if results["ollama"]["times"] and results["mlx"]["times"]:
            ollama_avg = statistics.mean(results["ollama"]["times"])
            mlx_avg = statistics.mean(results["mlx"]["times"])

            ollama_tps = statistics.mean(results["ollama"]["tokens_per_sec"])
            mlx_tps = statistics.mean(results["mlx"]["tokens_per_sec"])

            print(
                f"  • Inference Speed: {'Ollama' if ollama_avg < mlx_avg else 'MLX'} is {max(ollama_avg/mlx_avg, mlx_avg/ollama_avg):.1f}x faster"
            )
            print(
                f"  • Token Generation: {'Ollama' if ollama_tps > mlx_tps else 'MLX'} generates {max(ollama_tps/mlx_tps, mlx_tps/ollama_tps):.1f}x more tokens/sec"
            )
            print(f"  • Cold Start: Ollama wins (pre-loaded vs {mlx_load_time:.1f}s)")

        elif results["ollama"]["times"] and not results["mlx"]["times"]:
            print("  • Only Ollama completed successfully")
            print("  • MLX may have installation or model loading issues")
            print("  • Check if MLX is properly installed: pip install mlx-lm")
            print("  • Verify the MLX model is available")
            print(f"  • Try a smaller MLX model like 'mlx-community/Qwen2.5-3B-4bit'")
        elif not results["ollama"]["times"] and results["mlx"]["times"]:
            print("  • Only MLX completed successfully")
            print("  • Ollama may not be running or model not available")
            print("  • Start Ollama: ollama serve")
            print(f"  • Pull model: ollama pull {self.ollama_model}")
        else:
            print("  • Neither engine completed successfully")
            print("  • Check both Ollama and MLX installations")
            print("  • Verify models are available and compatible with your system")

        # Recommendations for different use cases
        print("\n💡 USE CASE RECOMMENDATIONS:")
        if results["ollama"]["times"] and results["mlx"]["times"]:
            if statistics.mean(results["ollama"]["times"]) < statistics.mean(
                results["mlx"]["times"]
            ):
                print("  • For production/server: Consider Ollama (faster inference)")
                print("  • For development/local: Both are viable")
            else:
                print("  • For production/server: Consider MLX (faster inference)")
                print("  • For development/local: Ollama (easier setup)")

        print("  • For memory-constrained systems: Use smaller models")
        print("  • For batch processing: Test with larger batches")

        # System recommendations
        print("\nSystem Optimization Tips:")
        if results["ollama"]["times"] or results["mlx"]["times"]:
            total_memory = psutil.virtual_memory().total / 1024 / 1024 / 1024  # GB
            print(f"  • System has {total_memory:.1f}GB RAM")
            if total_memory < 16:
                print("  • Consider using smaller models for systems with <16GB RAM")
            print("  • Close other applications to free up memory")
            print("  • Use GPU acceleration if available")


def main():
    test_prompts = [
        "Hello, how are you?",
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot learning to paint.",
    ]

    benchmark = FairInferenceBenchmark()
    results, load_time = benchmark.run_isolated_benchmark(test_prompts, iterations=2)
    benchmark.analyze_fair_results(results, load_time)

    # Cleanup
    print("\n🧹 Cleaning up...")
    benchmark.unload_mlx_model()
    benchmark.start_ollama()  # Restart Ollama for normal use


if __name__ == "__main__":
    main()
