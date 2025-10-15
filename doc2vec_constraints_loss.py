import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np
import gensim.downloader as api
from transformers import AutoTokenizer, AutoModel
import re
import random
import pickle
import os
import glob
import matplotlib.pyplot as plt
from itertools import combinations

class Doc2VecWithDynamicConstraints(nn.Module):
    def __init__(self, vocab_size, num_docs, embed_dim=300, pretrained_embeddings=None, 
                 freeze_embeddings=False, negative_samples=5, initW=0.5, initD=0.5):
        super().__init__()
        
        # Embeddings des mots de contexte (entrée)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Embeddings des mots de sortie (cible) - pour negative sampling
        self.output_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Si on a des embeddings pré-entraînés, on les charge
        if pretrained_embeddings is not None:
            self.context_embeddings.weight.data.copy_(torch.FloatTensor(pretrained_embeddings))
            self.output_embeddings.weight.data.copy_(torch.FloatTensor(pretrained_embeddings))
            
            if freeze_embeddings:
                self.context_embeddings.weight.requires_grad = False
        
        # Embeddings des documents (toujours entraînables)
        self.doc_embeddings = nn.Embedding(num_docs, embed_dim)
        
        # Paramètres pour negative sampling
        self.negative_samples = negative_samples
        self.vocab_size = vocab_size
        self.num_docs = num_docs
        
        # Poids de combinaison document/contexte
        self.alpha_doc = torch.tensor(0.5)
        self.alpha_context = torch.tensor(0.5)
        
        # Initialisation des embeddings
        nn.init.uniform_(self.doc_embeddings.weight, -initD/embed_dim, initD/embed_dim)
        
        if pretrained_embeddings is None:
            nn.init.uniform_(self.context_embeddings.weight, -initW/embed_dim, initW/embed_dim)
            nn.init.zeros_(self.output_embeddings.weight)
        
        self.embed_dim = embed_dim
        
        # Statistiques pour monitoring (seront mises à jour par le trainer)
        self.last_original_loss = 0.0
        self.last_constraint_loss = 0.0
    
    def freeze_word_embeddings(self):
        """Gèle les embeddings de mots pour l'inférence"""
        self.context_embeddings.weight.requires_grad = False
        self.output_embeddings.weight.requires_grad = False
        print("Embeddings de mots gelés pour l'inférence")
        
    def unfreeze_word_embeddings(self):
        """Dégèle les embeddings de mots"""
        self.context_embeddings.weight.requires_grad = True
        self.output_embeddings.weight.requires_grad = True
        print("Embeddings de mots dégelés")

    def freeze_context_word_embeddings(self):
        """Gèle les embeddings de mots W (context word embeddings uniquement) """
        self.context_embeddings.weight.requires_grad = False
        print("Embeddings de mots gelés W (context word embddings uniquement)")

    def unfreeze_context_word_embeddings(self):
        """Dégèle les embeddings de mots W (context word embeddings uniquement) """
        self.context_embeddings.weight.requires_grad = True
        print("Embeddings de mots dégelés W (context word embddings uniquement)")

    def freeze_doc_embeddings(self):
        """Gèle les embeddings de documents D """
        self.doc_embeddings.weight.requires_grad = False
        print("Embeddings de documents gelés")        

    def unfreeze_doc_embeddings(self):
        """Dégèle les embeddings de documents D """
        self.doc_embeddings.weight.requires_grad = True
        print("Embeddings de documents dégelés") 

    def resize_doc_embeddings(self, new_num_docs):
        """Redimensionne la matrice d'embeddings de documents pour l'inférence"""
        old_doc_embeddings = self.doc_embeddings.weight.data
        
        self.doc_embeddings = nn.Embedding(new_num_docs, self.embed_dim)
        nn.init.uniform_(self.doc_embeddings.weight, -0.5/self.embed_dim, 0.5/self.embed_dim)
        
        # Copier les anciens embeddings si ils existent
        min_docs = min(old_doc_embeddings.size(0), new_num_docs)
        if min_docs > 0:
            self.doc_embeddings.weight.data[:min_docs] = old_doc_embeddings[:min_docs]
        
        self.num_docs = new_num_docs
        print(f"Matrice d'embeddings de documents redimensionnée à {new_num_docs} documents")
        
    def resize_doc_embeddings_for_inference(self, new_num_docs):
        """Version spécifique pour l'inférence sans copie d'anciens embeddings"""
        self.doc_embeddings = nn.Embedding(new_num_docs, self.embed_dim)
        nn.init.uniform_(self.doc_embeddings.weight, -0.5/self.embed_dim, 0.5/self.embed_dim)
        self.num_docs = new_num_docs
        
    def forward(self, doc_ids, context_words, target_words, negative_words=None):
        """
        Forward pass standard sans contraintes (les contraintes sont gérées par le trainer)
        """
        batch_size = doc_ids.size(0)
        
        # 1. Embedding du document courant
        doc_embed = self.doc_embeddings(doc_ids)  # [batch_size, embed_dim]
        
        # 2. Embeddings des mots de contexte
        context_mask = (context_words != 0).float()  # [batch_size, context_size]
        context_embed = self.context_embeddings(context_words)  # [batch_size, context_size, embed_dim]

        # Application du masque et moyenne pondérée
        context_embed = context_embed * context_mask.unsqueeze(-1)
        context_lengths = context_mask.sum(dim=1, keepdim=True)
        context_lengths = torch.clamp(context_lengths, min=1)
        context_embed = torch.sum(context_embed, dim=1) / context_lengths  # [batch_size, embed_dim]
        
        # 3. Combinaison : document + contexte
        combined = self.alpha_doc * doc_embed + self.alpha_context * context_embed
        
        # 4. Calcul de la loss originale avec negative sampling
        target_embed = self.output_embeddings(target_words)  # [batch_size, embed_dim]
        
        # Score positif
        positive_score = torch.sum(combined * target_embed, dim=1)  # [batch_size]
        positive_loss = -torch.log(torch.sigmoid(positive_score))
        
        # Negative sampling
        if negative_words is not None:
            neg_embed = self.output_embeddings(negative_words)  # [batch_size, neg_samples, embed_dim]
            combined_expanded = combined.unsqueeze(1)  # [batch_size, 1, embed_dim]
            negative_scores = torch.sum(combined_expanded * neg_embed, dim=2)  # [batch_size, neg_samples]
            negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_scores)), dim=1)  # [batch_size]
            original_loss = positive_loss + negative_loss
        else:
            original_loss = positive_loss
        
        return original_loss.mean()
    
    def get_loss_components(self):
        """Retourne les composantes de la dernière perte calculée"""
        return {
            'original_loss': self.last_original_loss,
            'constraint_loss': self.last_constraint_loss
        }

class ConstraintManager:
    """Classe pour gérer les contraintes de manière flexible"""
    
    def __init__(self, constraint_matrix=None, constraint_lambda=0.1):
        self.constraint_matrix = constraint_matrix
        self.constraint_lambda = constraint_lambda
        self.constraint_pairs = None
        self.constraint_weights = None
        
        if constraint_matrix is not None:
            self._precompute_constraint_pairs()
    
    def set_constraints(self, constraint_matrix, constraint_lambda=None):
        """Définit une nouvelle matrice de contraintes"""
        self.constraint_matrix = constraint_matrix
        if constraint_lambda is not None:
            self.constraint_lambda = constraint_lambda
        
        if constraint_matrix is not None:
            self._precompute_constraint_pairs()
        else:
            self.constraint_pairs = None
            self.constraint_weights = None
    
    def _precompute_constraint_pairs(self):
        """Précalcule les paires de documents sous contrainte pour l'efficacité"""
        if self.constraint_matrix is None:
            self.constraint_pairs = None
            return
        
        constraint_matrix_tensor = torch.FloatTensor(self.constraint_matrix)
        
        # Trouve toutes les paires (i,j) où constraint_matrix[i,j] > 0
        constraint_indices = torch.nonzero(constraint_matrix_tensor, as_tuple=False)
        
        # Pour éviter la duplication (i,j) et (j,i), on garde seulement i < j
        pairs = []
        weights = []
        
        for idx in constraint_indices:
            i, j = idx[0].item(), idx[1].item()
            if i < j:  # Évite la duplication
                pairs.append([i, j])
                weights.append(constraint_matrix_tensor[i, j].item())
        
        if pairs:
            self.constraint_pairs = torch.LongTensor(pairs)
            self.constraint_weights = torch.FloatTensor(weights)
            print(f"Contraintes actives: {len(pairs)} paires de documents")
        else:
            self.constraint_pairs = None
            self.constraint_weights = None
            print("Aucune contrainte active trouvée")
    
    def compute_constraint_loss(self, doc_embeddings, batch_doc_ids=None):
        """
        Calcule le terme de perte des contraintes
        
        Args:
            doc_embeddings: matrice d'embeddings des documents [num_docs, embed_dim]
            batch_doc_ids: IDs des documents dans le batch actuel (optionnel)
                          Si fourni, calcule seulement les contraintes impliquant ces documents
        """
        if (self.constraint_pairs is None or 
            self.constraint_lambda == 0 or 
            self.constraint_matrix is None):
            return torch.tensor(0.0, device=doc_embeddings.device)
        
        device = doc_embeddings.device
        
        if batch_doc_ids is not None:
            # Mode efficace: calcule seulement les contraintes pertinentes pour le batch
            batch_set = set(batch_doc_ids.cpu().numpy())
            relevant_pairs = []
            relevant_weights = []
            
            for idx, (i, j) in enumerate(self.constraint_pairs):
                i_val, j_val = i.item(), j.item()
                if i_val in batch_set or j_val in batch_set:
                    relevant_pairs.append([i_val, j_val])
                    relevant_weights.append(self.constraint_weights[idx].item())
            
            if not relevant_pairs:
                return torch.tensor(0.0, device=device)
            
            pairs_tensor = torch.LongTensor(relevant_pairs).to(device)
            weights_tensor = torch.FloatTensor(relevant_weights).to(device)
        else:
            # Mode complet: toutes les contraintes
            pairs_tensor = self.constraint_pairs.to(device)
            weights_tensor = self.constraint_weights.to(device)
        
        # Calcule les distances euclidiennes au carré
        pairs_i = pairs_tensor[:, 0]  # [num_pairs]
        pairs_j = pairs_tensor[:, 1]  # [num_pairs]
        
        embeddings_i = doc_embeddings[pairs_i]  # [num_pairs, embed_dim]
        embeddings_j = doc_embeddings[pairs_j]  # [num_pairs, embed_dim]
        
        # Distance euclidienne au carré
        distances_sq = torch.sum((embeddings_i - embeddings_j) ** 2, dim=1)  # [num_pairs]
        
        # Moyenne pondérée des distances
        if weights_tensor is not None:
            constraint_loss = torch.sum(weights_tensor * distances_sq) / torch.sum(weights_tensor)
        else:
            constraint_loss = torch.mean(distances_sq)
        
        return constraint_loss
    
    def set_constraint_lambda(self, new_lambda):
        """Modifie le coefficient de régularisation des contraintes"""
        old_lambda = self.constraint_lambda
        self.constraint_lambda = new_lambda
        print(f"Coefficient de contraintes changé: {old_lambda:.4f} → {new_lambda:.4f}")
    
    def get_constraint_info(self):
        """Retourne des informations sur les contraintes"""
        if self.constraint_matrix is None:
            return "Aucune contrainte définie"
        
        num_constraints = len(self.constraint_pairs) if self.constraint_pairs is not None else 0
        total_possible = self.constraint_matrix.size if hasattr(self.constraint_matrix, 'size') else len(self.constraint_matrix) * len(self.constraint_matrix[0])
        
        return {
            'constraint_matrix_shape': np.array(self.constraint_matrix).shape,
            'active_constraints': num_constraints,
            'constraint_lambda': self.constraint_lambda,
            'constraint_density': num_constraints / (total_possible / 2) if total_possible > 0 else 0
        }
    
    def evaluate_constraint_satisfaction(self, doc_embeddings):
        """Évalue la satisfaction actuelle des contraintes"""
        if self.constraint_pairs is None:
            return None
        
        with torch.no_grad():
            constraint_loss = self.compute_constraint_loss(doc_embeddings)
            
            # Calcule les distances moyennes pour chaque contrainte
            device = doc_embeddings.device
            pairs_tensor = self.constraint_pairs.to(device)
            
            pairs_i = pairs_tensor[:, 0]
            pairs_j = pairs_tensor[:, 1]
            
            embeddings_i = doc_embeddings[pairs_i]
            embeddings_j = doc_embeddings[pairs_j]
            
            distances = torch.sqrt(torch.sum((embeddings_i - embeddings_j) ** 2, dim=1))
            
            return {
                'mean_constraint_distance': distances.mean().item(),
                'std_constraint_distance': distances.std().item(),
                'min_constraint_distance': distances.min().item(),
                'max_constraint_distance': distances.max().item(),
                'constraint_loss': constraint_loss.item()
            }
    
    def validate_constraint_matrix(self, num_docs):
        """Valide et ajuste la matrice de contraintes"""
        if self.constraint_matrix is None:
            return None
            
        if not isinstance(self.constraint_matrix, np.ndarray):
            self.constraint_matrix = np.array(self.constraint_matrix)
        
        if self.constraint_matrix.shape != (num_docs, num_docs):
            print(f"Ajustement de la matrice de contraintes: {self.constraint_matrix.shape} → {(num_docs, num_docs)}")
            if self.constraint_matrix.shape[0] > num_docs:
                self.constraint_matrix = self.constraint_matrix[:num_docs, :num_docs]
            else:
                new_matrix = np.zeros((num_docs, num_docs))
                min_size = min(self.constraint_matrix.shape[0], num_docs)
                new_matrix[:min_size, :min_size] = self.constraint_matrix[:min_size, :min_size]
                self.constraint_matrix = new_matrix
        
        # Assure la symétrie et supprime la diagonale
        self.constraint_matrix = (self.constraint_matrix > 0).astype(float)
        self.constraint_matrix = (self.constraint_matrix + self.constraint_matrix.T) > 0
        self.constraint_matrix = self.constraint_matrix.astype(float)
        np.fill_diagonal(self.constraint_matrix, 0)
        
        constraint_count = np.sum(self.constraint_matrix) / 2  # Diviser par 2 car matrice symétrique
        print(f"Matrice de contraintes validée: {int(constraint_count)} paires contraintes")
        
        # Recalculer les paires après validation
        self._precompute_constraint_pairs()
        
        return self.constraint_matrix

class EnhancedDoc2VecTrainerWithDynamicConstraints:
    """Version avec contraintes dynamiques définies au niveau du trainer"""
    
    def __init__(self, embedding_method='word2vec', model_name=None, embed_dim=300, 
                 window_size=5, negative_samples=5, min_count=1):
        self.embedding_method = embedding_method
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.min_count = min_count
        
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_counts = {}
        self.pretrained_embeddings = None
        self.transformer_model = None
        self.transformer_tokenizer = None
        self.negative_sampler = None
        self.trained_model = None
        
        # Gestionnaire de contraintes pour l'entraînement
        self.training_constraint_manager = None

    def train_doc2vec_with_constraints(self, documents, epochs=100, lr=0.01, batch_size=32,
                                     constraint_matrix=None, constraint_lambda=0.1,
                                     constraint_schedule=None, verbose_constraints=True,
                                     save_directory=None, save_at_epochs=None):
        """
        Entraîne le modèle Doc2Vec avec contraintes dynamiques
        
        Args:
            constraint_matrix: matrice de contraintes pour l'entraînement (peut être None)
            constraint_lambda: coefficient de régularisation pour les contraintes
            constraint_schedule: dict avec des epochs comme clés et lambdas comme valeurs
                                pour faire varier l'importance des contraintes au cours du temps
                                Exemple: {0: 0.0, 50: 0.05, 80: 0.1}
            save_directory: répertoire pour sauvegarder les checkpoints (créé si inexistant)
            save_at_epochs: liste des époques pour lesquelles sauvegarder un checkpoint
                           Exemple: [50, 100, 150]
        """
        # Importation locale pour éviter la duplication de code
        from doc2vec import NegativeSampler  # Assume la classe existe dans le fichier original
        
        # Configuration de la sauvegarde
        should_save_checkpoints = save_at_epochs is not None and save_directory is not None
        if should_save_checkpoints:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
                print(f"Répertoire de sauvegarde créé: {save_directory}")
            else:
                print(f"Utilisation du répertoire de sauvegarde: {save_directory}")
            
            # Convertir save_at_epochs en set pour des vérifications rapides
            save_epochs_set = set(save_at_epochs)
            print(f"Checkpoints programmés aux époques: {sorted(save_at_epochs)}")
        
        # Historique des pertes pour la sauvegarde
        loss_history = {
            'epochs': [],
            'total_loss': [],
            'original_loss': [],
            'constraint_loss': [],
            'constraint_lambda': []
        }
        
        vocab_size = self.build_vocabulary(documents)
        num_docs = len(documents)
        
        # Créer le gestionnaire de contraintes pour l'entraînement
        self.training_constraint_manager = ConstraintManager(constraint_matrix, constraint_lambda)
        if constraint_matrix is not None:
            self.training_constraint_manager.validate_constraint_matrix(num_docs)
        
        self.negative_sampler = NegativeSampler(vocab_size, self.word_counts)
        
        # Créer le modèle sans contraintes (les contraintes sont gérées par le trainer)
        model = Doc2VecWithDynamicConstraints(
            vocab_size=vocab_size,
            num_docs=num_docs,
            embed_dim=self.embed_dim,
            pretrained_embeddings=None,
            freeze_embeddings=False,
            negative_samples=self.negative_samples
        )
        
        training_data = self.generate_training_data(documents)
        
        if not training_data:
            print("Pas assez de données d'entraînement générées!")
            return None
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        print(f"\n=== ENTRAÎNEMENT AVEC CONTRAINTES DYNAMIQUES ===")
        if self.training_constraint_manager:
            print(f"Contraintes: {self.training_constraint_manager.get_constraint_info()}")
            print(f"Lambda initial: {constraint_lambda}")
        else:
            print("Aucune contrainte d'entraînement définie")
        
        model.train()
        for epoch in range(epochs):
            # Scheduler de contraintes optionnel
            if constraint_schedule and epoch in constraint_schedule:
                new_lambda = constraint_schedule[epoch]
                if self.training_constraint_manager:
                    self.training_constraint_manager.set_constraint_lambda(new_lambda)
            
            total_loss = 0
            num_batches = 0
            
            random.shuffle(training_data)
            
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                current_batch_size = len(batch)
                
                doc_ids = torch.LongTensor([item[0] for item in batch])
                contexts = [item[1] for item in batch]
                targets = torch.LongTensor([item[2] for item in batch])
                
                max_context_len = max(len(ctx) for ctx in contexts)
                if max_context_len == 0:
                    continue
                    
                padded_contexts = []
                for ctx in contexts:
                    padded = ctx + [0] * (max_context_len - len(ctx))
                    padded_contexts.append(padded)
                
                context_tensor = torch.LongTensor(padded_contexts)
                
                negative_samples_batch = []
                for j in range(current_batch_size):
                    target_word = targets[j].item()
                    context_words = [w for w in contexts[j] if w != 0]
                    positive_words = context_words + [target_word]
                    
                    neg_samples = self.negative_sampler.sample_negative_words(
                        positive_words, self.negative_samples
                    )
                    negative_samples_batch.append(neg_samples)
                
                negative_tensor = torch.LongTensor(negative_samples_batch)
                
                optimizer.zero_grad()
                
                # Calcul de la loss originale
                original_loss = model(doc_ids, context_tensor, targets, negative_tensor)
                
                # Calcul de la loss de contraintes si un gestionnaire existe
                constraint_loss = torch.tensor(0.0)
                if self.training_constraint_manager and self.training_constraint_manager.constraint_matrix is not None:
                    constraint_loss = self.training_constraint_manager.compute_constraint_loss(
                        model.doc_embeddings.weight, batch_doc_ids=doc_ids
                    )
                
                # Loss totale
                total_batch_loss = original_loss + self.training_constraint_manager.constraint_lambda * constraint_loss

                total_batch_loss.backward()
                optimizer.step()
                
                # Mise à jour des statistiques du modèle pour le monitoring
                model.last_original_loss = original_loss.item()
                model.last_constraint_loss = constraint_loss.item()
                
                total_loss += total_batch_loss.item()
                num_batches += 1
            
            # Calcul des métriques pour cette époque
            avg_loss = total_loss / max(num_batches, 1)
            
            # Sauvegarde de l'historique
            loss_history['epochs'].append(epoch)
            loss_history['total_loss'].append(avg_loss)
            loss_history['original_loss'].append(model.last_original_loss)
            loss_history['constraint_loss'].append(model.last_constraint_loss)
            loss_history['constraint_lambda'].append(
                self.training_constraint_manager.constraint_lambda if self.training_constraint_manager else 0.0
            )
            
            # Affichage périodique
            if epoch % 20 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch}/{epochs}:')
                print(f'  Loss totale: {avg_loss:.4f}')
                
                if verbose_constraints and self.training_constraint_manager:
                    print(f'  Loss originale: {model.last_original_loss:.4f}')
                    print(f'  Loss contraintes: {model.last_constraint_loss:.4f}')
                    constraint_term = self.training_constraint_manager.constraint_lambda * model.last_constraint_loss
                    print(f'  Terme contraintes: {constraint_term:.4f}')
                    
                    # Évaluation de la satisfaction des contraintes
                    if (epoch % 40 == 0 or epoch == epochs - 1) and self.training_constraint_manager.constraint_pairs is not None:
                        constraint_stats = self.training_constraint_manager.evaluate_constraint_satisfaction(
                            model.doc_embeddings.weight
                        )
                        if constraint_stats:
                            print(f'  Distance moyenne contraintes: {constraint_stats["mean_constraint_distance"]:.4f}')
            
            # Sauvegarde de checkpoint si nécessaire
            if should_save_checkpoints and epoch in save_epochs_set:
                self._save_checkpoint(model, epoch, save_directory, loss_history)
                print(f'  → Checkpoint sauvegardé à l\'époque {epoch}')
        
        print(f"\n=== ENTRAÎNEMENT TERMINÉ ===")
        
        # Évaluation finale des contraintes
        if self.training_constraint_manager and self.training_constraint_manager.constraint_pairs is not None:
            final_stats = self.training_constraint_manager.evaluate_constraint_satisfaction(
                model.doc_embeddings.weight
            )
            if final_stats:
                print(f"Satisfaction finale des contraintes:")
                for key, value in final_stats.items():
                    print(f"  {key}: {value:.4f}")
        
        self.trained_model = model
        
        model.eval()
        with torch.no_grad():
            doc_embeddings = model.doc_embeddings.weight.data.numpy()
            context_embeddings = model.context_embeddings.weight.data.numpy()
            output_embeddings = model.output_embeddings.weight.data.numpy()
        
        return doc_embeddings, context_embeddings, output_embeddings

    def resume_training(self, checkpoint_path, new_documents, additional_epochs=100, 
                                            output_adaptation_epochs=1000, lr=0.01, output_lr=0.001,
                                            batch_size=32, new_constraint_matrix=None, new_constraint_lambda=None,
                                            constraint_schedule=None, verbose_constraints=True,
                                            save_directory=None, save_at_epochs=None):
        """
        Fine-tuning simplifié en deux phases sur nouveaux documents uniquement :
        1. Entraînement conjoint (documents + mots) avec contraintes
        2. Adaptation spécialisée des embeddings de sortie
        
        Args:
            additional_epochs: nombre d'époques pour la phase 1 (défaut: 100)
            output_adaptation_epochs: nombre d'époques pour la phase 2 (défaut: 1000)
            output_lr: learning rate pour la phase 2 (défaut: 0.001, plus faible)
            new_constraint_matrix: matrice de contraintes [n_new_docs x n_new_docs]
            ... (autres paramètres identiques)
        
        Returns:
            tuple: (doc_embeddings, context_embeddings, output_embeddings)
        """
        from doc2vec import NegativeSampler
        
        # Charger le checkpoint pour récupérer les embeddings de mots pré-entraînés
        print(f"Chargement du checkpoint: {checkpoint_path}")
        model, metadata, previous_loss_history = self.load_checkpoint(checkpoint_path)
        
        previous_epochs = metadata['epoch']
        print(f"Modèle chargé - Époque: {previous_epochs}")
        print(f"Conservation des embeddings de mots, reset des embeddings de documents")
        
        # Redimensionner pour SEULEMENT les nouveaux documents
        num_new_docs = len(new_documents)
        print(f"Création d'une nouvelle matrice d'embeddings pour {num_new_docs} documents")
        
        # Remplacer complètement la matrice d'embeddings de documents
        model.doc_embeddings = nn.Embedding(num_new_docs, model.embed_dim)
        nn.init.uniform_(model.doc_embeddings.weight, -0.5/model.embed_dim, 0.5/model.embed_dim)
        model.num_docs = num_new_docs
        
        # Gérer les contraintes (simples car seulement nouveaux documents)
        if new_constraint_matrix is not None:
            constraint_matrix = np.array(new_constraint_matrix)
            constraint_lambda = new_constraint_lambda if new_constraint_lambda is not None else 0.1
            
            if constraint_matrix.shape == (num_new_docs, num_new_docs):
                print(f"Matrice de contraintes: {constraint_matrix.shape}")
                print(f"Nombre de contraintes actives: {np.sum(constraint_matrix > 0)}")
            else:
                print(f"ERREUR: Taille de matrice incorrecte: {constraint_matrix.shape}")
                print(f"Attendu: ({num_new_docs}, {num_new_docs})")
                constraint_matrix = None
                constraint_lambda = 0.0
        else:
            constraint_matrix = None
            constraint_lambda = 0.0
            print("Aucune contrainte définie")
        
        # Créer le gestionnaire de contraintes simplifié
        self.training_constraint_manager = ConstraintManager(constraint_matrix, constraint_lambda)
        if constraint_matrix is not None:
            self.training_constraint_manager.validate_constraint_matrix(num_new_docs)
        
        # Générer les données d'entraînement avec IDs commençant à 0
        new_training_data = self.generate_training_data(new_documents, start_doc_id=0)
        
        if not new_training_data:
            print("Pas assez de données d'entraînement générées!")
            return None
        
        print(f"Données d'entraînement générées pour {num_new_docs} nouveaux documents")
        print(f"Total paires d'entraînement: {len(new_training_data)}")
        
        # Configuration de la sauvegarde
        should_save_checkpoints = save_at_epochs is not None and save_directory is not None
        if should_save_checkpoints:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            save_epochs_set = set(save_at_epochs)
            print(f"Checkpoints programmés aux époques: {sorted(save_at_epochs)}")
        
        # Historique des pertes étendu pour les deux phases
        loss_history = {
            'epochs': [],
            'total_loss': [],
            'original_loss': [],
            'constraint_loss': [],
            'constraint_lambda': [],
            'phase': []  # Track la phase d'entraînement
        }
        
        print(f"\n=== FINE-TUNING SIMPLIFIÉ EN DEUX PHASES ===")
        print(f"Phase 1: {additional_epochs} époques d'entraînement conjoint")
        print(f"Phase 2: {output_adaptation_epochs} époques d'adaptation embeddings de sortie")
        print(f"Learning rates: Phase 1 = {lr}, Phase 2 = {output_lr}")
        print(f"Documents: {num_new_docs} nouveaux (anciens ignorés)")
        print("==============================================\n")
        
        model.train()
        start_epoch = previous_epochs + 1
        
        # ===============================
        # PHASE 1: ENTRAÎNEMENT CONJOINT
        # ===============================
        
        print(f">>> PHASE 1: ENTRAÎNEMENT CONJOINT <<<")
        print(f"Époques {start_epoch} à {start_epoch + additional_epochs - 1}")
        
        # Dégeler tous les embeddings pour la phase 1
        model.unfreeze_word_embeddings()
        model.doc_embeddings.weight.requires_grad = True
        
        # Optimiseur pour tous les paramètres
        optimizer_phase1 = optim.Adam(model.parameters(), lr=lr)
        
        phase1_end = start_epoch + additional_epochs
        
        for epoch in range(start_epoch, phase1_end):
            # Scheduler de contraintes optionnel
            if constraint_schedule and epoch in constraint_schedule:
                new_lambda = constraint_schedule[epoch]
                if self.training_constraint_manager:
                    self.training_constraint_manager.set_constraint_lambda(new_lambda)
            
            total_loss = 0
            num_batches = 0
            
            random.shuffle(new_training_data)
            
            for i in range(0, len(new_training_data), batch_size):
                batch = new_training_data[i:i+batch_size]
                current_batch_size = len(batch)
                
                doc_ids = torch.LongTensor([item[0] for item in batch])
                contexts = [item[1] for item in batch]
                targets = torch.LongTensor([item[2] for item in batch])
                
                max_context_len = max(len(ctx) for ctx in contexts)
                if max_context_len == 0:
                    continue
                    
                padded_contexts = []
                for ctx in contexts:
                    padded = ctx + [0] * (max_context_len - len(ctx))
                    padded_contexts.append(padded)
                
                context_tensor = torch.LongTensor(padded_contexts)
                
                # Negative sampling
                negative_samples_batch = []
                for j in range(current_batch_size):
                    target_word = targets[j].item()
                    context_words = [w for w in contexts[j] if w != 0]
                    positive_words = context_words + [target_word]
                    
                    neg_samples = self.negative_sampler.sample_negative_words(
                        positive_words, self.negative_samples
                    )
                    negative_samples_batch.append(neg_samples)
                
                negative_tensor = torch.LongTensor(negative_samples_batch)
                
                optimizer_phase1.zero_grad()
                
                # Calcul des losses
                original_loss = model(doc_ids, context_tensor, targets, negative_tensor)
                
                constraint_loss = torch.tensor(0.0)
                if (self.training_constraint_manager and 
                    self.training_constraint_manager.constraint_matrix is not None):
                    constraint_loss = self.training_constraint_manager.compute_constraint_loss(
                        model.doc_embeddings.weight, batch_doc_ids=doc_ids
                    )
                
                constraint_lambda_val = (self.training_constraint_manager.constraint_lambda 
                                if self.training_constraint_manager else 0.0)
                total_batch_loss = (original_loss + 
                            constraint_lambda_val * constraint_loss)
                
                total_batch_loss.backward()
                optimizer_phase1.step()
                
                model.last_original_loss = original_loss.item()
                model.last_constraint_loss = constraint_loss.item()
                
                total_loss += total_batch_loss.item()
                num_batches += 1
            
            # Enregistrement de l'historique Phase 1
            avg_loss = total_loss / max(num_batches, 1)
            loss_history['epochs'].append(epoch)
            loss_history['total_loss'].append(avg_loss)
            loss_history['original_loss'].append(model.last_original_loss)
            loss_history['constraint_loss'].append(model.last_constraint_loss)
            loss_history['constraint_lambda'].append(constraint_lambda_val)
            loss_history['phase'].append(1)
            
            # Affichage périodique Phase 1
            if (epoch - start_epoch) % 20 == 0 or epoch == phase1_end - 1:
                print(f'Phase 1 - Époque {epoch}/{phase1_end-1}:')
                print(f'  Loss totale: {avg_loss:.4f}')
                
                if verbose_constraints and self.training_constraint_manager:
                    print(f'  Loss originale: {model.last_original_loss:.4f}')
                    print(f'  Loss contraintes: {model.last_constraint_loss:.4f}')
            
            # Sauvegarde de checkpoint si nécessaire
            if should_save_checkpoints and epoch in save_epochs_set:
                metadata['epoch'] = epoch
                metadata['num_docs'] = num_new_docs
                metadata['training_phase'] = 1
                metadata['simplified_training'] = True
                self._save_checkpoint(model, epoch, save_directory, loss_history)
                print(f'  → Checkpoint Phase 1 sauvegardé à l\'époque {epoch}')
        
        print(f"Phase 1 terminée à l'époque {phase1_end - 1}")
        
        # ===============================================
        # PHASE 2: ADAPTATION DES EMBEDDINGS DE SORTIE
        # ===============================================
        
        print(f"\n>>> PHASE 2: ADAPTATION EMBEDDINGS DE SORTIE <<<")
        print(f"Époques {phase1_end} à {phase1_end + output_adaptation_epochs - 1}")
        
        # Geler les documents et les embeddings de contexte
        model.freeze_doc_embeddings()
        model.freeze_context_word_embeddings()
        
        print("Embeddings gelés:")
        print("  - Documents: GELÉS")
        print("  - Mots de contexte: GELÉS")
        print("  - Mots de sortie: ACTIFS")
        
        # Optimiseur seulement pour les embeddings de sortie
        optimizer_phase2 = optim.Adam(model.output_embeddings.parameters(), lr=output_lr)
        
        phase2_start = phase1_end
        phase2_end = phase2_start + output_adaptation_epochs
        
        for epoch in range(phase2_start, phase2_end):
            total_loss = 0
            num_batches = 0
            
            random.shuffle(new_training_data)
            
            for i in range(0, len(new_training_data), batch_size):
                batch = new_training_data[i:i+batch_size]
                current_batch_size = len(batch)
                
                doc_ids = torch.LongTensor([item[0] for item in batch])
                contexts = [item[1] for item in batch]
                targets = torch.LongTensor([item[2] for item in batch])
                
                max_context_len = max(len(ctx) for ctx in contexts)
                if max_context_len == 0:
                    continue
                    
                padded_contexts = []
                for ctx in contexts:
                    padded = ctx + [0] * (max_context_len - len(ctx))
                    padded_contexts.append(padded)
                
                context_tensor = torch.LongTensor(padded_contexts)
                
                # Negative sampling
                negative_samples_batch = []
                for j in range(current_batch_size):
                    target_word = targets[j].item()
                    context_words = [w for w in contexts[j] if w != 0]
                    positive_words = context_words + [target_word]
                    
                    neg_samples = self.negative_sampler.sample_negative_words(
                        positive_words, self.negative_samples
                    )
                    negative_samples_batch.append(neg_samples)
                
                negative_tensor = torch.LongTensor(negative_samples_batch)
                
                optimizer_phase2.zero_grad()
                
                # Phase 2: Seulement la loss originale, pas de contraintes
                original_loss = model(doc_ids, context_tensor, targets, negative_tensor)
                
                # En Phase 2, on optimise seulement pour la prédiction
                total_batch_loss = original_loss
                
                total_batch_loss.backward()
                optimizer_phase2.step()
                
                model.last_original_loss = original_loss.item()
                model.last_constraint_loss = 0.0  # Pas de contraintes en Phase 2
                
                total_loss += total_batch_loss.item()
                num_batches += 1
            
            # Enregistrement de l'historique Phase 2
            avg_loss = total_loss / max(num_batches, 1)
            loss_history['epochs'].append(epoch)
            loss_history['total_loss'].append(avg_loss)
            loss_history['original_loss'].append(model.last_original_loss)
            loss_history['constraint_loss'].append(0.0)  # Pas de contraintes
            loss_history['constraint_lambda'].append(0.0)
            loss_history['phase'].append(2)
            
            # Affichage périodique Phase 2
            if (epoch - phase2_start) % 100 == 0 or epoch == phase2_end - 1:
                print(f'Phase 2 - Époque {epoch}/{phase2_end-1}:')
                print(f'  Loss totale (originale): {avg_loss:.4f}')
            
            # Sauvegarde de checkpoint si nécessaire
            if should_save_checkpoints and epoch in save_epochs_set:
                metadata['epoch'] = epoch
                metadata['num_docs'] = num_new_docs
                metadata['training_phase'] = 2
                metadata['simplified_training'] = True
                self._save_checkpoint(model, epoch, save_directory, loss_history)
                print(f'  → Checkpoint Phase 2 sauvegardé à l\'époque {epoch}')
        
        print(f"Phase 2 terminée à l'époque {phase2_end - 1}")
        
        print(f"\n=== FINE-TUNING SIMPLIFIÉ DEUX PHASES TERMINÉ ===")
        print(f"Phase 1: {additional_epochs} époques d'entraînement conjoint")
        print(f"Phase 2: {output_adaptation_epochs} époques d'adaptation sortie")
        print(f"Total: {additional_epochs + output_adaptation_epochs} époques")
        print(f"Modèle final: {num_new_docs} documents, vocabulaire conservé")
        
        # Statistiques des phases
        phase1_epochs = sum(1 for p in loss_history['phase'] if p == 1)
        phase2_epochs = sum(1 for p in loss_history['phase'] if p == 2)
        print(f"Répartition: {phase1_epochs} époques Phase 1, {phase2_epochs} époques Phase 2")
        
        # Dégeler tout pour la suite
        model.unfreeze_word_embeddings()
        model.doc_embeddings.weight.requires_grad = True
        
        self.trained_model = model
        
        model.eval()
        with torch.no_grad():
            doc_embeddings = model.doc_embeddings.weight.data.numpy()
            context_embeddings = model.context_embeddings.weight.data.numpy()
            output_embeddings = model.output_embeddings.weight.data.numpy()
        
        return doc_embeddings, context_embeddings, output_embeddings
    
    def infer_new_documents(self, model_or_checkpoint, new_documents, 
                          inference_constraint_matrix=None, inference_constraint_lambda=0.1,
                          inference_strategy='adaptive', proportional_ratio=1.0, 
                          adaptive_epsilon=0.01, max_inference_epochs=200, 
                          fixed_epochs=50, inference_lr=0.001, inference_batch_size=32):
        """
        Inférence sur de nouveaux documents avec contraintes dynamiques spécifiques à l'inférence
        
        Args:
            model_or_checkpoint: modèle entraîné ou chemin vers un checkpoint
            new_documents: liste des nouveaux documents
            inference_constraint_matrix: matrice de contraintes spécifique à l'inférence (peut être différente de celle d'entraînement)
            inference_constraint_lambda: coefficient de régularisation pour l'inférence
            inference_strategy: 'fixed', 'adaptive' ou 'proportional'
            fixed_epochs: nombre d'époques fixe pour la stratégie 'fixed'
            ... (autres paramètres comme avant)
        
        Returns:
            dict contenant les embeddings inférés et l'historique des pertes
        """
        
        # Charger le modèle si nécessaire
        if isinstance(model_or_checkpoint, str):
            model, metadata, training_loss_history = self.load_checkpoint(model_or_checkpoint)
            target_loss = training_loss_history['total_loss'][-1] if training_loss_history else None
        else:
            model = model_or_checkpoint
            training_loss_history = None
            target_loss = None
        
        # Créer le gestionnaire de contraintes pour l'inférence
        inference_constraint_manager = None
        if inference_constraint_matrix is not None:
            inference_constraint_manager = ConstraintManager(
                inference_constraint_matrix, inference_constraint_lambda
            )
            inference_constraint_manager.validate_constraint_matrix(len(new_documents))
            print(f"Contraintes d'inférence: {inference_constraint_manager.get_constraint_info()}")
        else:
            print("Aucune contrainte d'inférence définie")
        
        # Redimensionner les embeddings de documents pour les nouveaux documents
        model.resize_doc_embeddings_for_inference(len(new_documents))
        model.freeze_word_embeddings()
        
        # Générer les données d'entraînement pour l'inférence
        training_data = self.generate_training_data(new_documents, start_doc_id=0)
        
        if not training_data:
            print("Pas assez de données pour l'inférence!")
            return None
        
        # Optimiseur seulement pour les embeddings de documents
        optimizer = optim.Adam(model.doc_embeddings.parameters(), lr=inference_lr)
        
        # Historique des pertes d'inférence
        inference_loss_history = {
            'epochs': [],
            'total_loss': [],
            'original_loss': [],
            'constraint_loss': [],
            'constraint_lambda': []
        }
        
        model.train()
        print(f"Inférence sur {len(new_documents)} nouveaux documents...")
        
        # Déterminer les paramètres selon la stratégie
        if inference_strategy == 'fixed':
            print(f"Stratégie fixed: {fixed_epochs} époques")
            
            for epoch in range(fixed_epochs):
                avg_loss = self._run_inference_epoch_with_constraints(
                    model, training_data, optimizer, inference_batch_size, 
                    inference_loss_history, epoch, inference_constraint_manager
                )
                
                if epoch % 10 == 0 or epoch == fixed_epochs - 1:
                    print(f'  Époque inférence {epoch}/{fixed_epochs}, Loss: {avg_loss:.4f}')
            
            early_stop = False  # Pour fixed, pas d'arrêt anticipé
            
        elif inference_strategy == 'adaptive':
            if target_loss is None:
                print("  Attention: pas de loss d'entraînement disponible, utilisation de 50 époques par défaut")
                max_epochs = 50
                target_loss = float('inf')
            else:
                max_epochs = max_inference_epochs
                print(f"Stratégie adaptive: target_loss={target_loss:.4f}, epsilon={adaptive_epsilon}")
            
            early_stop = False
            
            for epoch in range(max_epochs):
                avg_loss = self._run_inference_epoch_with_constraints(
                    model, training_data, optimizer, inference_batch_size, 
                    inference_loss_history, epoch, inference_constraint_manager
                )
                
                # Vérifier le critère d'arrêt adaptatif
                if target_loss != float('inf') and avg_loss <= (target_loss + adaptive_epsilon):
                    print(f'  Convergence atteinte à l\'époque {epoch}: '
                          f'loss={avg_loss:.4f} <= target+epsilon={target_loss + adaptive_epsilon:.4f}')
                    early_stop = True
                
                if epoch % 10 == 0 or epoch == max_epochs - 1 or early_stop:
                    print(f'  Époque inférence {epoch}/{max_epochs}, '
                          f'Loss: {avg_loss:.4f}, Target: {target_loss:.4f}')
                
                if early_stop:
                    break
            
            if not early_stop and target_loss != float('inf'):
                print(f'  Arrêt après {max_epochs} époques sans convergence complète')
        
        elif inference_strategy == 'proportional':
            # Calculer le nombre d'époques proportionnel (basé sur les époques d'entraînement)
            # Si on a accès aux métadonnées, utiliser le nombre d'époques d'entraînement
            if training_loss_history and 'epochs' in training_loss_history:
                training_epochs = len(training_loss_history['epochs'])
            else:
                training_epochs = 100  # Valeur par défaut
            
            max_epochs = int(proportional_ratio * training_epochs)
            max_epochs = max(1, max_epochs)  # Au moins 1 époque
            
            print(f"Stratégie proportional: ratio={proportional_ratio}, époques={max_epochs}")
            
            for epoch in range(max_epochs):
                avg_loss = self._run_inference_epoch_with_constraints(
                    model, training_data, optimizer, inference_batch_size, 
                    inference_loss_history, epoch, inference_constraint_manager
                )
                
                if epoch % 10 == 0 or epoch == max_epochs - 1:
                    print(f'  Époque inférence {epoch}/{max_epochs}, Loss: {avg_loss:.4f}')
            
            early_stop = False  # Pour proportional, pas d'arrêt anticipé
        
        else:
            raise ValueError(f"Stratégie inconnue: {inference_strategy}. Utilisez 'fixed', 'adaptive' ou 'proportional'")
        
        # Extraire les embeddings finaux
        model.eval()
        with torch.no_grad():
            inferred_embeddings = model.doc_embeddings.weight.data.numpy()
        
        final_loss = inference_loss_history['total_loss'][-1] if inference_loss_history['total_loss'] else 0.0
        
        return {
            'embeddings': inferred_embeddings,
            'loss_history': inference_loss_history,
            'num_documents': len(new_documents),
            'final_loss': final_loss,
            'strategy_used': inference_strategy,
            'inference_constraint_info': inference_constraint_manager.get_constraint_info() if inference_constraint_manager else None,
            'convergence_info': {
                'target_loss': target_loss if inference_strategy == 'adaptive' else None,
                'final_loss': final_loss,
                'converged': early_stop if inference_strategy == 'adaptive' else True,
                'total_epochs': len(inference_loss_history['epochs'])
            }
        } 
    
    def _run_inference_epoch_with_constraints(self, model, training_data, optimizer, batch_size, 
                                            loss_history, epoch, constraint_manager):
        """
        Exécute une époque d'inférence avec contraintes et met à jour l'historique
        
        Returns:
            float: loss moyenne pour cette époque
        """
        total_loss = 0
        num_batches = 0
        
        random.shuffle(training_data)
        
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]
            current_batch_size = len(batch)
            
            doc_ids = torch.LongTensor([item[0] for item in batch])
            contexts = [item[1] for item in batch]
            targets = torch.LongTensor([item[2] for item in batch])
            
            max_context_len = max(len(ctx) for ctx in contexts)
            if max_context_len == 0:
                continue
                
            padded_contexts = []
            for ctx in contexts:
                padded = ctx + [0] * (max_context_len - len(ctx))
                padded_contexts.append(padded)
            
            context_tensor = torch.LongTensor(padded_contexts)
            
            # Negative sampling
            negative_samples_batch = []
            for j in range(current_batch_size):
                target_word = targets[j].item()
                context_words = [w for w in contexts[j] if w != 0]
                positive_words = context_words + [target_word]
                
                neg_samples = self.negative_sampler.sample_negative_words(
                    positive_words, self.negative_samples
                )
                negative_samples_batch.append(neg_samples)
            
            negative_tensor = torch.LongTensor(negative_samples_batch)
            
            optimizer.zero_grad()
            
            # Calcul de la loss originale
            original_loss = model(doc_ids, context_tensor, targets, negative_tensor)
            
            # Calcul de la loss de contraintes d'inférence si un gestionnaire existe
            constraint_loss = torch.tensor(0.0)
            if constraint_manager and constraint_manager.constraint_matrix is not None:
                constraint_loss = constraint_manager.compute_constraint_loss(
                    model.doc_embeddings.weight, batch_doc_ids=doc_ids
                )
            
            # Loss totale
            constraint_lambda = constraint_manager.constraint_lambda if constraint_manager else 0.0
            total_batch_loss = original_loss + constraint_lambda * constraint_loss
            
            total_batch_loss.backward()
            optimizer.step()
            
            # Mise à jour des statistiques du modèle pour le monitoring
            model.last_original_loss = original_loss.item()
            model.last_constraint_loss = constraint_loss.item()
            
            total_loss += total_batch_loss.item()
            num_batches += 1
        
        # Calculer la loss moyenne et enregistrer les métriques
        avg_loss = total_loss / max(num_batches, 1)
        
        loss_history['epochs'].append(epoch)
        loss_history['total_loss'].append(avg_loss)
        loss_history['original_loss'].append(model.last_original_loss)
        loss_history['constraint_loss'].append(model.last_constraint_loss)
        loss_history['constraint_lambda'].append(constraint_lambda)
        
        return avg_loss
    
    def _save_checkpoint(self, model, epoch, save_directory, loss_history):
        """
        Sauvegarde un checkpoint du modèle à une époque donnée
        
        Args:
            model: le modèle à sauvegarder
            epoch: numéro de l'époque actuelle
            save_directory: répertoire de sauvegarde
            loss_history: historique des pertes jusqu'à cette époque
        """
        # Noms des fichiers
        base_name = f"checkpoint_epoch_{epoch}"
        model_path = os.path.join(save_directory, f"{base_name}.pth")
        metadata_path = os.path.join(save_directory, f"{base_name}_metadata.pkl")
        loss_history_path = os.path.join(save_directory, f"{base_name}_loss_history.npy")
        
        # Sauvegarde du modèle
        torch.save(model.state_dict(), model_path)
        
        # Sauvegarde des métadonnées
        metadata = {
            'epoch': epoch,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'word_counts': self.word_counts,
            'vocab_size': len(self.word_to_idx),
            'num_docs': model.num_docs,
            'embed_dim': self.embed_dim,
            'window_size': self.window_size,
            'negative_samples': self.negative_samples,
            'min_count': self.min_count,
            'embedding_method': self.embedding_method,
            'model_name': self.model_name,
            'training_constraint_matrix': self.training_constraint_manager.constraint_matrix if self.training_constraint_manager else None,
            'training_constraint_lambda': self.training_constraint_manager.constraint_lambda if self.training_constraint_manager else 0.0,
            'training_constraint_info': self.training_constraint_manager.get_constraint_info() if self.training_constraint_manager else None
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Sauvegarde de l'historique des pertes
        # Copie l'historique jusqu'à l'époque actuelle (incluse)
        current_loss_history = {}
        for key, values in loss_history.items():
            current_loss_history[key] = values.copy()
        
        np.save(loss_history_path, current_loss_history)
        
        print(f"    Fichiers sauvegardés:")
        print(f"      - Modèle: {model_path}")
        print(f"      - Métadonnées: {metadata_path}")
        print(f"      - Historique: {loss_history_path}")
    
    def load_checkpoint(self, checkpoint_path, metadata_path=None, loss_history_path=None):
        """
        Charge un checkpoint sauvegardé
        
        Args:
            checkpoint_path: chemin vers le fichier .pth du modèle
            metadata_path: chemin vers le fichier .pkl des métadonnées (optionnel, inféré si None)
            loss_history_path: chemin vers le fichier .npy de l'historique (optionnel, inféré si None)
        
        Returns:
            tuple: (model, metadata, loss_history)
        """
        # Inférer les chemins des fichiers associés si non fournis
        if metadata_path is None:
            metadata_path = checkpoint_path.replace('.pth', '_metadata.pkl')
        if loss_history_path is None:
            loss_history_path = checkpoint_path.replace('.pth', '_loss_history.npy')
        
        # Charger les métadonnées
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Restaurer les attributs du trainer
        self.word_to_idx = metadata['word_to_idx']
        self.idx_to_word = metadata['idx_to_word']
        self.word_counts = metadata['word_counts']
        self.embed_dim = metadata['embed_dim']
        self.window_size = metadata['window_size']
        self.negative_samples = metadata['negative_samples']
        self.min_count = metadata['min_count']
        self.embedding_method = metadata['embedding_method']
        self.model_name = metadata.get('model_name')
        
        # Restaurer le gestionnaire de contraintes d'entraînement si disponible
        training_constraint_matrix = metadata.get('training_constraint_matrix')
        training_constraint_lambda = metadata.get('training_constraint_lambda', 0.1)
        if training_constraint_matrix is not None:
            self.training_constraint_manager = ConstraintManager(
                training_constraint_matrix, training_constraint_lambda
            )
        else:
            self.training_constraint_manager = None
        
        # Créer le modèle
        model = Doc2VecWithDynamicConstraints(
            vocab_size=metadata['vocab_size'],
            num_docs=metadata['num_docs'],
            embed_dim=metadata['embed_dim'],
            negative_samples=metadata['negative_samples']
        )
        
        # Charger les poids du modèle
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        
        # Charger l'historique des pertes si disponible
        loss_history = None
        if os.path.exists(loss_history_path):
            loss_history = np.load(loss_history_path, allow_pickle=True).item()
        
        # Recréer le negative sampler
        from doc2vec import NegativeSampler
        self.negative_sampler = NegativeSampler(metadata['vocab_size'], self.word_counts)
        
        self.trained_model = model
        
        print(f"Checkpoint chargé:")
        print(f"  Époque: {metadata['epoch']}")
        print(f"  Vocabulaire: {metadata['vocab_size']} mots")
        print(f"  Documents: {metadata['num_docs']}")
        print(f"  Contraintes d'entraînement: {metadata.get('training_constraint_info', 'Non disponible')}")
        
        return model, metadata, loss_history
    
    def batch_infer_from_checkpoints(self, checkpoint_directory, new_documents, 
                                   output_directory, inference_constraint_matrix=None,
                                   inference_constraint_lambda=0.1, inference_strategy='adaptive',
                                   proportional_ratio=1.0, adaptive_epsilon=0.01,
                                   max_inference_epochs=200, fixed_epochs=50, 
                                   inference_lr=0.001, inference_batch_size=32, create_plots=True):
        """
        Lance l'inférence sur de nouveaux documents à partir de tous les checkpoints disponibles
        avec contraintes d'inférence dynamiques
        
        Args:
            checkpoint_directory: répertoire contenant les checkpoints
            new_documents: liste des nouveaux documents
            output_directory: répertoire de sortie (créé si inexistant)
            inference_constraint_matrix: matrice de contraintes spécifique à l'inférence
            inference_constraint_lambda: coefficient de régularisation pour l'inférence
            inference_strategy: 'fixed', 'adaptive' ou 'proportional'
            fixed_epochs: nombre d'époques fixe pour la stratégie 'fixed'
            ... (autres paramètres comme avant)
        
        Returns:
            dict: résultats d'inférence pour chaque checkpoint
        """
        
        # Créer le répertoire de sortie
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            print(f"Répertoire de sortie créé: {output_directory}")
        
        # Trouver tous les checkpoints
        checkpoint_pattern = os.path.join(checkpoint_directory, "checkpoint_epoch_*.pth")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            print(f"Aucun checkpoint trouvé dans {checkpoint_directory}")
            return {}
        
        # Extraire les numéros d'époque et trier
        checkpoint_info = []
        for checkpoint_file in checkpoint_files:
            basename = os.path.basename(checkpoint_file)
            epoch_str = basename.replace("checkpoint_epoch_", "").replace(".pth", "")
            try:
                epoch_num = int(epoch_str)
                checkpoint_info.append((epoch_num, checkpoint_file))
            except ValueError:
                print(f"Ignore le fichier avec format invalide: {basename}")
        
        checkpoint_info.sort()  # Trier par époque
        print(f"Checkpoints trouvés: {len(checkpoint_info)} fichiers")
        print(f"Époques disponibles: {[info[0] for info in checkpoint_info]}")
        
        if inference_constraint_matrix is not None:
            print(f"Contraintes d'inférence définies: matrice {np.array(inference_constraint_matrix).shape}")
        else:
            print("Aucune contrainte d'inférence définie")
        
        results = {}
        
        for epoch_num, checkpoint_file in checkpoint_info:
            print(f"\n=== INFÉRENCE DEPUIS CHECKPOINT ÉPOQUE {epoch_num} ===")
            
            try:
                # Inférence avec contraintes dynamiques
                inference_results = self.infer_new_documents(
                    checkpoint_file, new_documents,
                    inference_constraint_matrix=inference_constraint_matrix,
                    inference_constraint_lambda=inference_constraint_lambda,
                    inference_strategy=inference_strategy,
                    proportional_ratio=proportional_ratio,
                    adaptive_epsilon=adaptive_epsilon,
                    max_inference_epochs=max_inference_epochs,
                    fixed_epochs=fixed_epochs,
                    inference_lr=inference_lr,
                    inference_batch_size=inference_batch_size
                )
                
                if inference_results is None:
                    print(f"Échec de l'inférence pour l'époque {epoch_num}")
                    continue
                
                # Créer le répertoire de sortie pour cette époque
                epoch_output_dir = os.path.join(output_directory, f"epoch_{epoch_num}")
                if not os.path.exists(epoch_output_dir):
                    os.makedirs(epoch_output_dir)
                
                # Charger les métadonnées du checkpoint pour la sauvegarde
                _, metadata, training_loss_history = self.load_checkpoint(checkpoint_file)
                
                # Sauvegarder les résultats
                self._save_inference_results(
                    inference_results, epoch_output_dir, epoch_num, metadata,
                    training_loss_history, inference_strategy, create_plots
                )
                
                results[epoch_num] = {
                    'inference_results': inference_results,
                    'metadata': metadata,
                    'output_directory': epoch_output_dir
                }
                
                print(f"Inférence terminée pour l'époque {epoch_num}")
                print(f"Résultats sauvegardés dans: {epoch_output_dir}")
                
            except Exception as e:
                print(f"Erreur lors de l'inférence pour l'époque {epoch_num}: {str(e)}")
                continue
        
        print(f"\n=== INFÉRENCE BATCH TERMINÉE ===")
        print(f"Résultats disponibles pour {len(results)} checkpoints")
        
        return results
    
    def _save_inference_results(self, inference_results, output_dir, checkpoint_epoch, 
                              metadata, training_loss_history, inference_strategy, create_plots):
        """
        Sauvegarde les résultats d'inférence pour un checkpoint
        """
        if inference_results is None:
            print(f"Pas de résultats à sauvegarder pour l'époque {checkpoint_epoch}")
            return
        
        # Fichiers de sortie
        embeddings_path = os.path.join(output_dir, "inferred_embeddings.npy")
        loss_history_path = os.path.join(output_dir, "inference_loss_history.npy")
        metadata_path = os.path.join(output_dir, "inference_metadata.pkl")
        plot_path = os.path.join(output_dir, "convergence_plot.png")
        
        # Sauvegarder les embeddings
        np.save(embeddings_path, inference_results['embeddings'])
        
        # Sauvegarder l'historique des pertes d'inférence
        np.save(loss_history_path, inference_results['loss_history'])
        
        # Sauvegarder les métadonnées complètes
        inference_metadata = {
            'checkpoint_epoch': checkpoint_epoch,
            'inference_strategy': inference_strategy,
            'num_inferred_documents': inference_results['num_documents'],
            'final_inference_loss': inference_results['final_loss'],
            'original_metadata': metadata,
            'embeddings_shape': inference_results['embeddings'].shape,
            'inference_epochs': len(inference_results['loss_history']['epochs']),
            'strategy_used': inference_results['strategy_used'],
            'convergence_info': inference_results['convergence_info'],
            'inference_constraint_info': inference_results.get('inference_constraint_info')
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(inference_metadata, f)
        
        # Créer le graphique de convergence
        if create_plots:
            try:
                self._create_convergence_plot(
                    inference_results['loss_history'], 
                    training_loss_history,
                    checkpoint_epoch, 
                    plot_path
                )
            except Exception as e:
                print(f"Erreur lors de la création du graphique: {str(e)}")
        
        print(f"    Fichiers sauvegardés:")
        print(f"      - Embeddings: {embeddings_path}")
        print(f"      - Historique inférence: {loss_history_path}")
        print(f"      - Métadonnées: {metadata_path}")
        if create_plots and os.path.exists(plot_path):
            print(f"      - Graphique: {plot_path}")
    
    def _create_convergence_plot(self, inference_loss_history, training_loss_history, 
                               checkpoint_epoch, plot_path):
        """
        Crée un graphique de convergence combinant entraînement et inférence
        """
        plt.figure(figsize=(14, 10))
        
        # Graphique principal: perte totale avec ligne de convergence
        plt.subplot(2, 3, 1)
        if training_loss_history and 'total_loss' in training_loss_history:
            plt.plot(training_loss_history['epochs'], training_loss_history['total_loss'], 
                    'b-', label='Entraînement', alpha=0.7)
            target_loss = training_loss_history['total_loss'][-1]
            plt.axhline(y=target_loss, color='orange', linestyle=':', alpha=0.8, 
                       label=f'Target Loss ({target_loss:.4f})')
            plt.axvline(x=checkpoint_epoch, color='r', linestyle='--', alpha=0.5, 
                       label=f'Checkpoint (époque {checkpoint_epoch})')
        
        plt.plot(inference_loss_history['epochs'], inference_loss_history['total_loss'], 
                'g-', label='Inférence', linewidth=2)
        plt.title('Perte Totale')
        plt.xlabel('Époques')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Décomposition des pertes d'inférence
        plt.subplot(2, 3, 2)
        plt.plot(inference_loss_history['epochs'], inference_loss_history['original_loss'], 
                'b-', label='Loss Originale', alpha=0.8)
        plt.plot(inference_loss_history['epochs'], inference_loss_history['constraint_loss'], 
                'r-', label='Loss Contraintes', alpha=0.8)
        plt.title('Décomposition des Pertes (Inférence)')
        plt.xlabel('Époques')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Lambda des contraintes
        plt.subplot(2, 3, 3)
        plt.plot(inference_loss_history['epochs'], inference_loss_history['constraint_lambda'], 
                'purple', label='Lambda Contraintes', linewidth=2)
        plt.title('Coefficient des Contraintes')
        plt.xlabel('Époques')
        plt.ylabel('Lambda')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Comparaison finale
        plt.subplot(2, 3, 4)
        final_losses = {
            'Original': inference_loss_history['original_loss'][-1],
            'Contraintes': inference_loss_history['constraint_loss'][-1],
            'Total': inference_loss_history['total_loss'][-1]
        }
        
        bars = plt.bar(final_losses.keys(), final_losses.values(), 
                      color=['blue', 'red', 'green'], alpha=0.7)
        plt.title('Pertes Finales (Inférence)')
        plt.ylabel('Loss')
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, final_losses.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.4f}', ha='center', va='bottom')
        
        # Évolution de la loss d'inférence avec zone de convergence
        plt.subplot(2, 3, 5)
        plt.plot(inference_loss_history['epochs'], inference_loss_history['total_loss'], 
                'g-', linewidth=2, label='Loss Inférence')
        
        if training_loss_history and 'total_loss' in training_loss_history:
            target_loss = training_loss_history['total_loss'][-1]
            plt.axhline(y=target_loss, color='orange', linestyle=':', 
                       label=f'Target ({target_loss:.4f})')
            # Zone de convergence (target ± epsilon)
            epsilon = 0.01  # Valeur par défaut
            plt.fill_between(inference_loss_history['epochs'], 
                           target_loss - epsilon, target_loss + epsilon, 
                           alpha=0.2, color='orange', label='Zone convergence (±ε)')
        
        plt.title('Convergence vers Target')
        plt.xlabel('Époques')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Métadonnées de convergence
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # Texte avec informations de convergence
        info_text = f"Checkpoint Époque: {checkpoint_epoch}\n"
        info_text += f"Époques d'inférence: {len(inference_loss_history['epochs'])}\n"
        info_text += f"Loss finale: {inference_loss_history['total_loss'][-1]:.4f}\n"
        
        if training_loss_history and 'total_loss' in training_loss_history:
            target_loss = training_loss_history['total_loss'][-1]
            final_loss = inference_loss_history['total_loss'][-1]
            info_text += f"Target loss: {target_loss:.4f}\n"
            info_text += f"Écart: {abs(final_loss - target_loss):.4f}\n"
            
            if final_loss <= target_loss + 0.01:
                info_text += "✓ Convergence atteinte"
            else:
                info_text += "✗ Convergence non atteinte"
        
        plt.text(0.1, 0.7, info_text, fontsize=10, family='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        plt.title('Informations de Convergence')
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # Fermer pour libérer la mémoire
    
    # Méthodes utilitaires (reprises du code original avec adaptations)
    def preprocess_text(self, text):
        """Préprocessing du texte"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        words = text.split()
        words = [word for word in words if len(word) > 2]
        return words
    
    def build_vocabulary(self, documents):
        """Construit le vocabulaire à partir des documents"""
        word_counts = Counter()
        
        for doc in documents:
            words = self.preprocess_text(doc)
            word_counts.update(words)
        
        vocab = [word for word, count in word_counts.items() if count >= self.min_count]
        
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        self.word_counts = {self.word_to_idx[word]: count 
                          for word, count in word_counts.items() 
                          if word in self.word_to_idx}
        
        print(f"Vocabulaire construit: {len(vocab)} mots uniques")
        print(f"Mots les plus fréquents: {word_counts.most_common(10)}")
        
        return len(vocab)
    
    def generate_training_data(self, documents, start_doc_id=0):
        """Génère les données d'entraînement pour Doc2Vec avec negative sampling"""
        training_data = []
        
        for doc_idx, doc in enumerate(documents):
            doc_id = start_doc_id + doc_idx
            words = self.preprocess_text(doc)
            word_indices = [self.word_to_idx.get(word, -1) for word in words]
            word_indices = [idx for idx in word_indices if idx != -1]
            
            if len(word_indices) < self.window_size + 1:
                continue
            
            for i in range(self.window_size, len(word_indices) - self.window_size):
                context_start = max(0, i - self.window_size)
                context_end = min(len(word_indices), i + self.window_size + 1)
                context = word_indices[context_start:i] + word_indices[i+1:context_end]
                
                if len(context) > 0:
                    training_data.append((doc_id, context, word_indices[i]))
        
        print(f"Total paires d'entraînement générées: {len(training_data)}")
        return training_data
     
def should_freeze_doc_embeddings(epoch, freeze_schedule=None):
    """
    Détermine si les embeddings de documents doivent être gelés à cette époque
    
    Args:
        epoch: époque actuelle
        freeze_schedule: dict ou list définissant les plages de gel
                        Exemples:
                        - [(0, 50), (100, 150)] : gel époques 0-50 et 100-150
                        - {0: True, 50: False, 100: True, 150: False} : transitions
    """
    if freeze_schedule is None:
        return False
        
    if isinstance(freeze_schedule, list):
        # Format liste de tuples (start, end)
        for start, end in freeze_schedule:
            if start <= epoch <= end:
                return True
        return False
    
    elif isinstance(freeze_schedule, dict):
        # Format dict avec époques de transition
        applicable_epochs = [e for e in freeze_schedule.keys() if e <= epoch]
        if applicable_epochs:
            latest_epoch = max(applicable_epochs)
            return freeze_schedule[latest_epoch]
        return False
    
    return False