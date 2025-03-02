LinguisticProbabilisticExperiment.py
import torch
import numpy as np
from tqdm import tqdm  # For progress bars
import pandas as pd
from scipy.stats import entropy
from transformers import AutoTokenizer
# Assuming LinguisticQuantumPerturber is in perturber.py
from perturber import LinguisticQuantumPerturber

class LinguisticProbabilisticExperiment:
    def __init__(self, perturber, model_name="distilbert-base-uncased"):
        """
        Initializes the experiment class.

        Args:
            perturber (LinguisticQuantumPerturber): An instance of the perturbation class.
        """
        self.perturber = perturber
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) #Required for KL Divergence
        self.model = perturber.model # Access the model from the perturber
        self.results = []  # Store experimental results

    def load_data(self, data_source, max_sentences=None):
        """
        Loads sentences from a data source.

        Args:
            data_source (str or list):  Either a path to a text file (one sentence per line)
                                        or a list of strings (sentences).
            max_sentences (int, optional):  Maximum number of sentences to load.

        Returns:
            list: A list of sentences (strings).
        """
        sentences = []
        if isinstance(data_source, str):  # File path
            try:
                with open(data_source, 'r', encoding='utf-8') as f:
                    for line in f:
                        sentence = line.strip()
                        if sentence:  # Skip empty lines
                            sentences.append(sentence)
                            if max_sentences is not None and len(sentences) >= max_sentences:
                                break
            except FileNotFoundError:
                print(f"Error: File not found: {data_source}")
                return []  # Return empty list on error

        elif isinstance(data_source, list):  # List of sentences
            sentences = data_source[:max_sentences] if max_sentences is not None else data_source
        else:
            print("Error: Invalid data_source type. Must be a file path (str) or a list of sentences.")
            return []

        return sentences
    
    def _calculate_kl_divergence(self, original_logits, perturbed_logits):
        """Calculates KL divergence between two probability distributions."""

        # Convert logits to probabilities using softmax
        original_probs = torch.nn.functional.softmax(original_logits, dim=-1)
        perturbed_probs = torch.nn.functional.softmax(perturbed_logits, dim=-1)

        # Calculate KL divergence.  Use a small epsilon for numerical stability.
        kl_div = entropy(original_probs.cpu().numpy(), perturbed_probs.cpu().numpy() + 1e-12)
        return kl_div
    
    def _get_model_output(self, sentence):
        """Gets model outputs (logits and attention) for a sentence."""
        inputs = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=True, truncation=True, padding=True).to(self.perturber.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        return outputs.logits, outputs.attentions

    def run_experiment(self, sentences, perturbation_types, num_perturbations=1, similarity_threshold=0.2, frequency_threshold_factor=2.0, batch_size=32):
        """
        Runs the perturbation experiment.

        Args:
            sentences (list): A list of sentences (strings).
            perturbation_types (list): List of perturbation types (e.g., ['synonym', 'antonym']).
            num_perturbations (int): Number of words to perturb per sentence.
            batch_size (int): The number of sentences to process at a time
        """
        self.results = []  # Clear previous results
        
        for i in tqdm(range(0, len(sentences), batch_size), desc="Processing sentences"):
            batch_sentences = sentences[i:i+batch_size]

            # Get original model outputs for the entire batch
            original_logits_batch = []
            original_attentions_batch = []
            for sentence in batch_sentences:
                original_logits, original_attentions = self._get_model_output(sentence)
                original_logits_batch.append(original_logits)
                original_attentions_batch.append(original_attentions)  # Store attention for later

            # Now, perturb each sentence in the batch
            for j, original_sentence in enumerate(batch_sentences):
                original_logits = original_logits_batch[j]  # Logits for the current sentence

                for perturbation_type in perturbation_types:
                    perturbed_sentences = self.perturber.perturb_sentence(
                        original_sentence,
                        perturbation_type,
                        num_perturbations=num_perturbations,
                        similarity_threshold=similarity_threshold,
                        frequency_threshold_factor=frequency_threshold_factor
                    )

                    for perturbed_sentence in perturbed_sentences:
                        # Get model output for the perturbed sentence
                        perturbed_logits, perturbed_attentions = self._get_model_output(perturbed_sentence)

                        # Calculate KL Divergence
                        kl_divergence = self._calculate_kl_divergence(original_logits.squeeze(0), perturbed_logits.squeeze(0))  # Remove batch dimension

                        # Calculate Semantic Similarity
                        semantic_similarity = self.perturber.compute_semantic_similarity(original_sentence, perturbed_sentence)

                        # Store Results
                        result = {
                            'original_sentence': original_sentence,
                            'perturbed_sentence': perturbed_sentence,
                            'perturbation_type': perturbation_type,
                            'kl_divergence': kl_divergence,
                            'semantic_similarity': semantic_similarity,
                            # You'll add attention shift calculations here later
                        }
                        self.results.append(result)


    def get_results_dataframe(self):
        """Returns the results as a Pandas DataFrame."""
        return pd.DataFrame(self.results)

    def analyze_results(self):
        """Provides basic statistical analysis of the results."""
        if not self.results:
            print("No results to analyze. Run an experiment first.")
            return

        df = self.get_results_dataframe()

        # Group by perturbation type and calculate mean/std for KL divergence and similarity
        grouped_results = df.groupby('perturbation_type').agg({
            'kl_divergence': ['mean', 'std', 'count'],
            'semantic_similarity': ['mean', 'std']
        })
        print(grouped_results)
        return grouped_results

# Example Usage
if __name__ == '__main__':
    # Create a Perturber instance (you can customize its parameters)
    perturber = LinguisticQuantumPerturber()

    # Create an Experiment instance
    experiment = LinguisticProbabilisticExperiment(perturber)

    # Load some sentences (replace with your data source)
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "This is an example sentence for testing.",
        "Artificial intelligence is a fascinating field.",
        "The cat sat on the mat.",
        "She sells seashells by the seashore."
    ]
    #Or load from file
    #sentences = experiment.load_data("sentences.txt", max_sentences=100)

    # Run the experiment
    experiment.run_experiment(sentences, perturbation_types=['synonym', 'antonym', 'related', 'random', 'mlm'], num_perturbations=1)

    # Get the results as a DataFrame
    results_df = experiment.get_results_dataframe()
    print(results_df.head())

    # Analyze the results
    experiment.analyze_results()
    
    #Save to csv
    results_df.to_csv("experiment_results.csv", index=False)
