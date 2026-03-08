from modules.QnA_Pipeline import QNA_pipeline, QNA_prep
from data.keywords import KEYWORDS as CUSTUM_KEYWORDS
import os
import pandas as pd
import json

def model_prediction(question_):
    HF_TOKEN = ""
    init = False
    if init:
        PREP = QNA_prep("data", CUSTUM_KEYWORDS)
    Qna_dataframe = pd.read_csv("data/qna_dataframe.csv")
    with open(f'data/all_keywords.json', 'r') as file:
                keywords = json.load(file)
    all_keywords = keywords[0]['all_keywords']
    main_keywords = keywords[1]['main_keywords']

    FAQ_pipeline = QNA_pipeline(HF_TOKEN, Qna_dataframe, CUSTUM_KEYWORDS, all_keywords, main_keywords)
    result = FAQ_pipeline.predition_pipeline(question_)
    print('############################################# \n \n \n \n #########################################')
    print(result)
    print('############################################# \n \n \n \n #########################################')



    from huggingface_hub import InferenceClient

    HF_TOKEN = ""

    client = InferenceClient(token=HF_TOKEN)  # ✅ pas de provider forcé




    # --- normaliser les keywords pour l'affichage ---


    score = float(result.get("similarity_score", 0))

    messages = [
        {
            "role": "system",
            "content": """
                Tu es un assistant d’information pour une collectivité locale. 
                Tu dois reformuler une réponse officielle 
                Tu DOIS repondre UNIQUEMENT en français.
                Ne mentionne pas la FAQ
                
            """
        },
        {
            "role": "user",
            "content": f"""
    Voici la question d’un utilisateur :
    {result['question']}

    Voici les mots-clés standards associés à cette question (contexte) :
    {result['standard_keyword']}

    Voici une réponse officielle issue d’une base FAQ :
    {result["chosen_answer"]}

    Voici le degré de confiance (0 à 100) indiquant à quel point cette réponse est adaptée :
    {score}

    Règles :
    - Si le score < 30 : demande à l’utilisateur de préciser ou reformuler sa question.
    - Si le score < 10 : reformule brièvement la question puis indique que tu ne peux pas répondre.
    - Si le score >= 30 : reformule la réponse officielle.

    Contraintes :
    - Reste strictement fidèle au contenu fourni.
    - N’invente aucune information (coordonnées, contacts, démarches non mentionnées) mais si elle est présente donne là.
    - Ton clair, simple et neutre.

    Réponds uniquement avec le texte final destiné à l’utilisateur.
    """
        }
    ]

    try:
        response = client.chat_completion(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=messages,
            max_tokens=250,
            temperature=0.2,
        )
        final_answer = response.choices[0].message["content"].strip()

    except Exception as e:
        # ✅ fallback si chat pas dispo (ou réseau instable)
        print("⚠️ chat_completion indisponible, fallback text_generation :", e)

        prompt = messages[0]["content"] + "\n\n" + messages[1]["content"]
        final_answer = client.text_generation(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            prompt=prompt,
            max_new_tokens=250,
            temperature=0.2,
        ).strip()

    print("############################################")
    print(result["question"])
    print("---------------------------------------")
    print(final_answer)

    


model_prediction("Je viens d'avoir un bébé et je ne suis pas mariée avec le père, quelles démarches dois-je faire ?")