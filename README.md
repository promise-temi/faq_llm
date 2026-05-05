# Contexte du projet

Dans le cadre de la transformation numérique de la collectivité Val de Loire Numérique, ce projet vise à automatiser la réponse aux questions fréquentes des citoyens.

Actuellement, les agents passent une grande partie de leur temps à traiter des demandes récurrentes (état civil, urbanisme, transport, etc.). L’objectif est donc de développer un assistant intelligent accessible via une API REST.

# Objectif principal

Concevoir, développer et déployer une API FAQ intégrant un modèle de langage (LLM), en s’appuyant sur une démarche de benchmark rigoureuse pour identifier la meilleure stratégie de réponse.

# Objectifs spécifiques
Benchmark : comparer plusieurs approches de réponse automatique
Recommandation : sélectionner la meilleure stratégie basée sur des métriques objectives
Implémentation : développer une API robuste
Industrialisation : mettre en place tests automatisés et pipeline CI/CD
Documentation : fournir une documentation technique exploitable

# Architecture du projet

Le système repose sur un pipeline  :

- Recherche de réponse (FAQ matching)
- Utilisation d’un pipeline QNA_pipeline avec un matching basé sur mots-clés et similarité.
- Évaluation de pertinence
- Calcul d’un score de similarité permettant d’évaluer la qualité de la correspondance.
- Génération de réponse (LLM)
- Reformulation de la réponse via un modèle de langage tout en restant fidèle aux données sources.
- Pipeline de réponse

L’utilisateur pose une question
Le système identifie les mots-clés et recherche la meilleure réponse dans la FAQ
Un score de similarité est calculé

# Le modèle applique des règles de décision :
score < 10 : impossibilité de répondre
score < 30 : demande de clarification
score ≥ 30 : reformulation de la réponse

Démarche de benchmark





# Modèles utilisés
Modèle de langage : mistralai/Mistral-7B-Instruct
Modèle d’embeddings : intfloat/multilingual-e5-large
API : HuggingFace Inference API

# Industrialisation
- Architecture modulaire
- Dataset structuré et versionné
- Export des résultats de benchmark au format parquet
- Préparation pour intégration dans un pipeline CI/CD
  
# Améliorations possibles
- Intégration d’un moteur de recherche vectoriel (RAG)
- Fine-tuning du modèle
- Mise en place d’un système de feedback utilisateur
- Ajout de monitoring en production
- Choix technique et recommandation

La stratégie retenue repose sur une approche hybride combinant un matching FAQ et un modèle de langage pour la reformulation.
Cette approche permet de garantir la fiabilité des réponses, d’éviter les hallucinations du modèle, tout en conservant une bonne qualité de restitution et une adaptabilité aux besoins métier.
