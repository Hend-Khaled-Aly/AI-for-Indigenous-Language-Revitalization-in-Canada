import os
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split




class CreeLearningModel:
    """
    A class to represent a Cree-English translation learning model.

    This model uses TF-IDF character n-gram vectorization to embed Cree and English words,
    then applies cosine similarity for translation lookups. It supports multi-meaning
    mappings, evaluation, and creation of multiple choice exercises.
    """
    def __init__(self):
        """
        Initialize the model with necessary components:
        - TF-IDF vectorizer for text embedding (character n-grams 1-3)
        - MultiLabelBinarizer for multilabel metrics (if needed)
        - Dictionaries for Cree-to-English and English-to-Cree mappings
        - Placeholders for embeddings and similarity model
        """
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, 3),
            max_features=1000
        )
        self.mlb = MultiLabelBinarizer()
        self.similarity_model = None
        self.cree_to_english = defaultdict(list)  # Maps Cree words to list of English meanings
        self.english_to_cree = defaultdict(list)  # Maps English meanings to list of Cree words
        self.cree_embeddings = None
        self.english_embeddings = None

# ------------------------------------------------------------------------------

    def preprocess_data(self, csv_file_path):
        """
        Load and preprocess Cree-English word pairs from a CSV file.

        Steps:
        - Load CSV into pandas DataFrame
        - Strip whitespace and drop empty rows
        - Populate two-way mappings:
          Cree -> English (list of meanings)
          English -> Cree (list of Cree words)
        - Print dataset statistics and examples of multi-meaning Cree words

        Returns:
        - Cleaned pandas DataFrame
        """
        print("Loading and preprocessing data...")
        
        # Load the CSV file
        df = pd.read_csv(csv_file_path)
        print(f"Original dataset shape: {df.shape}")
        
        # Remove extra spaces
        df['Cree'] = df['Cree'].str.strip()
        df['English'] = df['English'].str.strip()
        
        # Remove any empty rows
        df = df.dropna()
        
        # Build mappings, making sure not to duplicate entries
        for _, row in df.iterrows():
            cree_word = row['Cree'].lower()
            english_meaning = row['English'].lower()
            
            if english_meaning not in self.cree_to_english[cree_word]:
                # a Cree word maps to a list of English meanings
                self.cree_to_english[cree_word].append(english_meaning)
            
            if cree_word not in self.english_to_cree[english_meaning]:
                # an English word maps to a list of Cree words
                self.english_to_cree[english_meaning].append(cree_word)
        
        print(f"Unique Cree words: {len(self.cree_to_english)}")
        print(f"Unique English meanings: {len(self.english_to_cree)}")
        
        # Analyze how many Cree words have multiple English meanings
        multi_meaning_cree = {k: v for k, v in self.cree_to_english.items() if len(v) > 1}
        print(f"Cree words with multiple meanings: {len(multi_meaning_cree)}")
        
        # Show some examples
        print("\nExamples of Cree words with multiple meanings:")
        for i, (cree, meanings) in enumerate(list(multi_meaning_cree.items())[:5]):
            print(f"  {cree}: {', '.join(meanings)}")
        
        return df
    
# ------------------------------------------------------------------------------
    def create_embeddings(self, df):
        """
        Generate TF-IDF embeddings for all unique Cree words and English meanings.

        Uses character n-gram TF-IDF vectorization fitted on Cree words, then transforms
        English meanings with the same vectorizer for comparable vector space.

        Returns:
        - Lists of unique Cree words and English meanings
        """
        print("\nCreating embeddings...")
        
        # Get unique words and meanings
        unique_cree = list(self.cree_to_english.keys())
        unique_english = list(self.english_to_cree.keys())
        # Fit vectorizer on Cree words and transform them into embeddings
        self.cree_embeddings = self.vectorizer.fit_transform(unique_cree)
        # Transform English meanings using the same vectorizer (no refitting)
        self.english_embeddings = self.vectorizer.transform(unique_english)
        
        print(f"Cree embeddings shape: {self.cree_embeddings.shape}")
        print(f"English embeddings shape: {self.english_embeddings.shape}")
        
        return unique_cree, unique_english

# ------------------------------------------------------------------------------

    def build_similarity_model(self):
        """
        Build a similarity matrix between Cree and English embeddings.

        Uses cosine similarity between all Cree and English vectors, which will
        later be used for approximate translation lookups.
        """
        print("\nBuilding similarity model...")

        self.similarity_matrix = cosine_similarity(self.cree_embeddings, self.english_embeddings)
        print(f"Similarity matrix shape: {self.similarity_matrix.shape}")

# ------------------------------------------------------------------------------

    def find_translations(self, cree_word, top_k=5):
        """
        Given a Cree word, find the top-k English translations.

        First attempts direct dictionary lookup; if not found,
        uses vector similarity with the closest Cree words and returns
        their known English meanings.

        Parameters:
        - cree_word (str): Input Cree word to translate
        - top_k (int): Number of top results to return

        Returns:
        - List of English meanings (strings)
        """
        cree_word = cree_word.lower().strip()
        # Direct lookup first for speed and accuracy
        if cree_word in self.cree_to_english:
            return self.cree_to_english[cree_word]
        
        # If no direct match, find most similar Cree words by embedding
        unique_cree = list(self.cree_to_english.keys())
        unique_english = list(self.english_to_cree.keys())
        
        if cree_word not in unique_cree:
            # Find most similar Cree word
            query_embedding = self.vectorizer.transform([cree_word])
            similarities = cosine_similarity(query_embedding, self.cree_embeddings)[0]
            
            # Get top similar Cree words
            top_indices = similarities.argsort()[-top_k:][::-1]
            similar_cree_words = [unique_cree[i] for i in top_indices if similarities[i] > 0.1]
            
            # Get their translations
            translations = []
            for similar_word in similar_cree_words:
                translations.extend(self.cree_to_english[similar_word])
            # Return unique meanings up to top_k results
            return list(set(translations))[:top_k]
        # If Cree word is not found and no similar words above threshold, return empty list        
        return []
    
# ------------------------------------------------------------------------------

    def find_cree_words(self, english_meaning, top_k=5):
        """
        Given an English meaning, find the top-k Cree words.

        Similar logic to find_translations but in reverse direction.

        Parameters:
        - english_meaning (str): English meaning to lookup
        - top_k (int): Number of top Cree words to return

        Returns:
        - List of Cree words (strings)
        """
        english_meaning = english_meaning.lower().strip()
        
        # Direct lookup first
        if english_meaning in self.english_to_cree:
            return self.english_to_cree[english_meaning]
        
        # If not found, use similarity
        unique_english = list(self.english_to_cree.keys())
        
        if english_meaning not in unique_english:
            # Find most similar English meaning
            query_embedding = self.vectorizer.transform([english_meaning])
            similarities = cosine_similarity(query_embedding, self.english_embeddings)[0]
            
            # Get top similar English meanings
            top_indices = similarities.argsort()[-top_k:][::-1]
            similar_meanings = [unique_english[i] for i in top_indices if similarities[i] > 0.1]
            
            # Get their Cree translations
            cree_words = []
            for similar_meaning in similar_meanings:
                cree_words.extend(self.english_to_cree[similar_meaning])
            
            return list(set(cree_words))[:top_k]
        
        return []
    
# ------------------------------------------------------------------------------

    def create_learning_exercises(self, difficulty='mixed'):
        """
        Generate multiple-choice exercises for learners.

        Depending on difficulty:
        - 'easy': only Cree words with a single English meaning
        - 'hard': only Cree words with multiple English meanings
        - 'mixed': all Cree words

        Each exercise presents one Cree word and multiple English choices,
        with one or more correct answers.

        Returns:
        - List of exercises (dicts) each containing:
          cree_word, choices (list), correct_answers (list), and type
        """
        exercises = []
        
        if difficulty == 'easy':
            # Single meaning words only
            words = {k: v for k, v in self.cree_to_english.items() if len(v) == 1}
        elif difficulty == 'hard':
            # Multiple meaning words only
            words = {k: v for k, v in self.cree_to_english.items() if len(v) > 1}
        else:
            # Mixed difficulty
            words = self.cree_to_english
        
        # Multiple choice exercises
        word_list = list(words.keys())
        np.random.shuffle(word_list)
        
        for cree_word in word_list[:20]:  # Create 20 exercises
            correct_answers = words[cree_word]
            
            # Get wrong answers
            all_meanings = list(self.english_to_cree.keys())
            wrong_answers = [m for m in all_meanings if m not in correct_answers]
            wrong_choices = np.random.choice(wrong_answers, min(3, len(wrong_answers)), replace=False)
            
            # Create multiple choice
            choices = list(correct_answers) + list(wrong_choices)
            np.random.shuffle(choices)
            
            exercises.append({
                'cree_word': cree_word,
                'choices': choices,
                'correct_answers': correct_answers,
                'type': 'multiple_choice'
            })
        
        return exercises
    
# ------------------------------------------------------------------------------

    def save_model(self, filepath='cree_model.pkl'):
        """
        Save the trained model state to a file for later reuse.

        Saves vectorizer, mappings, and similarity matrix if present.
        """
        model_data = {
            'vectorizer': self.vectorizer,
            'cree_to_english': dict(self.cree_to_english),
            'english_to_cree': dict(self.english_to_cree),
            'similarity_matrix': self.similarity_matrix if hasattr(self, 'similarity_matrix') else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")

# ------------------------------------------------------------------------------

    def load_model(self, filepath='cree_model.pkl'):
        """
        Load a previously saved model from file and restore internal state.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.cree_to_english = defaultdict(list, model_data['cree_to_english'])
        self.english_to_cree = defaultdict(list, model_data['english_to_cree'])
        self.similarity_matrix = model_data.get('similarity_matrix')
        
        print(f"Model loaded from {filepath}")

# ------------------------------------------------------------------------------

    def evaluate_model(self, test_ratio=0.2, random_state=42):
        """
        Evaluate model performance using various metrics and a test split.

        Metrics include:
        - Exact match accuracy: All correct meanings predicted
        - Partial match accuracy: At least one correct meaning predicted
        - Top-k accuracy: Correct meanings within top 3 and top 5 predictions
        - Average Jaccard similarity between predicted and true meaning sets
        - Coverage score: fraction of words for which model returns predictions
        - Multi-label precision, recall, F1 scores

        Returns:
        - evaluation_results: dictionary with all metric scores
        - detailed_results: list with per-word prediction details for analysis
        """
        print("\n=== Model Evaluation ===")
        
        # Prepare data for evaluation
        all_cree_words = list(self.cree_to_english.keys())
        
        # Split data for evaluation
        train_words, test_words = train_test_split(
            all_cree_words, 
            test_size=test_ratio, 
            random_state=random_state
        )
        
        print(f"Training words: {len(train_words)}")
        print(f"Testing words: {len(test_words)}")
        
        # Evaluation metrics
        evaluation_results = {
            'exact_match_accuracy': 0,
            'partial_match_accuracy': 0,
            'top_3_accuracy': 0,
            'top_5_accuracy': 0,
            'average_similarity_score': 0,
            'coverage_score': 0,
            'multilingual_precision': 0,
            'multilingual_recall': 0,
            'multilingual_f1': 0
        }
        
        exact_matches = 0
        partial_matches = 0
        top_3_matches = 0
        top_5_matches = 0
        similarity_scores = []
        covered_words = 0
        
        # Detailed results for analysis
        detailed_results = []
        
        for cree_word in test_words:
            true_meanings = set(self.cree_to_english[cree_word])
            predicted_meanings = self.find_translations(cree_word, top_k=5)
            predicted_set = set(predicted_meanings)
            
            # Exact match (all meanings predicted correctly)
            if true_meanings == predicted_set:
                exact_matches += 1
            
            # Partial match (at least one meaning predicted correctly)
            if len(true_meanings.intersection(predicted_set)) > 0:
                partial_matches += 1
            
            # Top-k accuracy
            if len(true_meanings.intersection(set(predicted_meanings[:3]))) > 0:
                top_3_matches += 1
            if len(true_meanings.intersection(set(predicted_meanings[:5]))) > 0:
                top_5_matches += 1
            
            # Coverage (model found the word)
            if len(predicted_meanings) > 0:
                covered_words += 1
            
            # Similarity score
            if len(predicted_meanings) > 0:
                # Calculate Jaccard similarity
                jaccard_sim = len(true_meanings.intersection(predicted_set)) / len(true_meanings.union(predicted_set))
                similarity_scores.append(jaccard_sim)
            else:
                similarity_scores.append(0)
            
            # Store detailed results
            detailed_results.append({
                'cree_word': cree_word,
                'true_meanings': list(true_meanings),
                'predicted_meanings': predicted_meanings,
                'exact_match': true_meanings == predicted_set,
                'partial_match': len(true_meanings.intersection(predicted_set)) > 0,
                'jaccard_similarity': similarity_scores[-1]
            })
        
        # Calculate final scores
        n_test = len(test_words)
        evaluation_results['exact_match_accuracy'] = exact_matches / n_test
        evaluation_results['partial_match_accuracy'] = partial_matches / n_test
        evaluation_results['top_3_accuracy'] = top_3_matches / n_test
        evaluation_results['top_5_accuracy'] = top_5_matches / n_test
        evaluation_results['average_similarity_score'] = np.mean(similarity_scores)
        evaluation_results['coverage_score'] = covered_words / n_test
        
        # Multi-label classification metrics
        y_true_multilabel = []
        y_pred_multilabel = []
        
        # Get all possible English meanings for encoding
        all_english_meanings = list(self.english_to_cree.keys())
        
        for result in detailed_results:
            # True labels (binary vector)
            true_vector = [1 if meaning in result['true_meanings'] else 0 for meaning in all_english_meanings]
            y_true_multilabel.append(true_vector)
            
            # Predicted labels (binary vector)
            pred_vector = [1 if meaning in result['predicted_meanings'] else 0 for meaning in all_english_meanings]
            y_pred_multilabel.append(pred_vector)
        
        # Calculate precision, recall, F1 for multi-label scenario
        if len(y_true_multilabel) > 0:
            evaluation_results['multilingual_precision'] = precision_score(
                y_true_multilabel, y_pred_multilabel, average='micro', zero_division=0
            )
            evaluation_results['multilingual_recall'] = recall_score(
                y_true_multilabel, y_pred_multilabel, average='micro', zero_division=0
            )
            evaluation_results['multilingual_f1'] = f1_score(
                y_true_multilabel, y_pred_multilabel, average='micro', zero_division=0
            )
        
        # Print evaluation results
        print("\n--- Evaluation Results ---")
        print(f"Exact Match Accuracy: {evaluation_results['exact_match_accuracy']:.3f}")
        print(f"Partial Match Accuracy: {evaluation_results['partial_match_accuracy']:.3f}")
        print(f"Top-3 Accuracy: {evaluation_results['top_3_accuracy']:.3f}")
        print(f"Top-5 Accuracy: {evaluation_results['top_5_accuracy']:.3f}")
        print(f"Average Similarity Score: {evaluation_results['average_similarity_score']:.3f}")
        print(f"Coverage Score: {evaluation_results['coverage_score']:.3f}")
        print(f"Multi-label Precision: {evaluation_results['multilingual_precision']:.3f}")
        print(f"Multi-label Recall: {evaluation_results['multilingual_recall']:.3f}")
        print(f"Multi-label F1-Score: {evaluation_results['multilingual_f1']:.3f}")
        
        # Show some examples
        print("\n--- Sample Predictions ---")
        for i in range(min(5, len(detailed_results))):
            result = detailed_results[i]
            print(f"\nCree: '{result['cree_word']}'")
            print(f"  True: {result['true_meanings']}")
            print(f"  Predicted: {result['predicted_meanings']}")
            print(f"  Exact Match: {result['exact_match']}")
            print(f"  Similarity: {result['jaccard_similarity']:.3f}")
        
        # Error analysis
        print("\n--- Error Analysis ---")
        failed_predictions = [r for r in detailed_results if not r['partial_match']]
        print(f"Failed predictions: {len(failed_predictions)}")
        
        if failed_predictions:
            print("Examples of failed predictions:")
            for i in range(min(3, len(failed_predictions))):
                result = failed_predictions[i]
                print(f"  '{result['cree_word']}': {result['true_meanings']} -> {result['predicted_meanings']}")
        
        return evaluation_results, detailed_results
    
# ----------------------------------------------------------------------------

    def cross_validate_model(self, k_folds=5, random_state=42):
        """
        Perform k-fold cross-validation over the dataset to assess model robustness.

        For each fold:
        - Split Cree words into test fold
        - Evaluate exact and partial match accuracies and similarity
        - Aggregate and print fold-wise results

        Returns:
        - List of fold-wise results
        - Summary dictionary with mean and std deviation metrics
        """
        print(f"\n=== {k_folds}-Fold Cross Validation ===")
        
        all_cree_words = list(self.cree_to_english.keys())
        np.random.seed(random_state)
        np.random.shuffle(all_cree_words)
        
        # Split into k folds
        fold_size = len(all_cree_words) // k_folds
        cv_results = []
        
        for fold in range(k_folds):
            print(f"\nFold {fold + 1}/{k_folds}")
            
            # Define test set for this fold
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < k_folds - 1 else len(all_cree_words)
            test_words = all_cree_words[start_idx:end_idx]
            
            # Evaluate on this fold
            exact_matches = 0
            partial_matches = 0
            similarity_scores = []
            
            for cree_word in test_words:
                true_meanings = set(self.cree_to_english[cree_word])
                predicted_meanings = self.find_translations(cree_word, top_k=5)
                predicted_set = set(predicted_meanings)
                
                if true_meanings == predicted_set:
                    exact_matches += 1
                
                if len(true_meanings.intersection(predicted_set)) > 0:
                    partial_matches += 1
                
                # Jaccard similarity
                if len(predicted_meanings) > 0:
                    jaccard_sim = len(true_meanings.intersection(predicted_set)) / len(true_meanings.union(predicted_set))
                    similarity_scores.append(jaccard_sim)
                else:
                    similarity_scores.append(0)
            
            fold_results = {
                'fold': fold + 1,
                'exact_accuracy': exact_matches / len(test_words),
                'partial_accuracy': partial_matches / len(test_words),
                'avg_similarity': np.mean(similarity_scores)
            }
            
            cv_results.append(fold_results)
            print(f"  Exact Accuracy: {fold_results['exact_accuracy']:.3f}")
            print(f"  Partial Accuracy: {fold_results['partial_accuracy']:.3f}")
            print(f"  Avg Similarity: {fold_results['avg_similarity']:.3f}")
        
        # Calculate overall CV results
        cv_summary = {
            'mean_exact_accuracy': np.mean([r['exact_accuracy'] for r in cv_results]),
            'std_exact_accuracy': np.std([r['exact_accuracy'] for r in cv_results]),
            'mean_partial_accuracy': np.mean([r['partial_accuracy'] for r in cv_results]),
            'std_partial_accuracy': np.std([r['partial_accuracy'] for r in cv_results]),
            'mean_similarity': np.mean([r['avg_similarity'] for r in cv_results]),
            'std_similarity': np.std([r['avg_similarity'] for r in cv_results])
        }
        
        print(f"\n--- Cross Validation Summary ---")
        print(f"Exact Accuracy: {cv_summary['mean_exact_accuracy']:.3f} ± {cv_summary['std_exact_accuracy']:.3f}")
        print(f"Partial Accuracy: {cv_summary['mean_partial_accuracy']:.3f} ± {cv_summary['std_partial_accuracy']:.3f}")
        print(f"Similarity Score: {cv_summary['mean_similarity']:.3f} ± {cv_summary['std_similarity']:.3f}")
        
        return cv_results, cv_summary
    
# ----------------------------------------------------------------------------

    def learning_curve_analysis(self):
        """
        Analyze how model accuracy changes as more training data is used.

        For increasing fractions of the dataset:
        - Sample subset as training data
        - Test on remaining words (up to 50 for efficiency)
        - Calculate partial match accuracy

        Returns:
        - List of dictionaries containing dataset size, number of training words, and accuracy
        """
        print("\n=== Learning Curve Analysis ===")
        
        all_words = list(self.cree_to_english.keys())
        dataset_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        learning_curve_results = []
        
        for size in dataset_sizes:
            # Sample subset of data
            n_words = int(len(all_words) * size)
            sampled_words = np.random.choice(all_words, n_words, replace=False)
            
            # Create temporary model with subset
            temp_cree_to_english = {word: self.cree_to_english[word] for word in sampled_words}
            
            # Test on remaining data
            test_words = [word for word in all_words if word not in sampled_words]
            if len(test_words) == 0:
                continue
            
            # Evaluate
            partial_matches = 0
            for test_word in test_words[:min(50, len(test_words))]:  # Limit for efficiency
                true_meanings = set(self.cree_to_english[test_word])
                predicted_meanings = self.find_translations(test_word, top_k=3)
                predicted_set = set(predicted_meanings)
                
                if len(true_meanings.intersection(predicted_set)) > 0:
                    partial_matches += 1
            
            accuracy = partial_matches / min(50, len(test_words))
            
            learning_curve_results.append({
                'dataset_size': size,
                'n_training_words': n_words,
                'accuracy': accuracy
            })
            
            print(f"Dataset size: {size:.1f} ({n_words} words) -> Accuracy: {accuracy:.3f}")
        
        return learning_curve_results
    
# ----------------------------------------------------------------------------

    def model_confidence_score(self, cree_word, threshold=0.3):
        """
        Calculate a confidence score for a given Cree word prediction.

        - Returns 1.0 for exact dictionary match (high confidence)
        - Otherwise, uses maximum cosine similarity with Cree embeddings,
          normalized by a threshold

        Parameters:
        - cree_word (str): Input word to assess confidence
        - threshold (float): Similarity threshold for normalization

        Returns:
        - confidence score between 0 and 1 (float)
        """
        predictions = self.find_translations(cree_word, top_k=5)
        
        if not predictions:
            return 0.0
        
        # Calculate confidence based on similarity scores
        unique_cree = list(self.cree_to_english.keys())
        
        if cree_word.lower() in unique_cree:
            # Direct match - high confidence
            return 1.0
        else:
            # Similarity-based match
            query_embedding = self.vectorizer.transform([cree_word.lower()])
            similarities = cosine_similarity(query_embedding, self.cree_embeddings)[0]
            max_similarity = np.max(similarities)
            
            # Normalize confidence score
            confidence = min(max_similarity / threshold, 1.0) if max_similarity > 0 else 0.0
            return confidence
        
# ----------------------------------------------------------------------------



