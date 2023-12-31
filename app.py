# Importation des librairies.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import langchain

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain

# Recuperation de la valeur de l'API KEY
import yaml

with open("conf.yml") as fh:
    config = yaml.safe_load(fh)

OPENAI_API_KEY = config["API_KEY"]["VALUE"]

# Ajout de l'api key dans l'environnemnt.
import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Creation de l'application avec FastAPI().
app = FastAPI()

# Choix du modele OpenAI

# Text-davinci-003 est le modèle qui a le score le plus élevé sur les cotes de préférence humaine.
# Son entrainement génère généralement un résultat moins toxique et plus véridique.
# Ce modèle propose également de nouvelles fonctionnalités permettant de mieux gérer les instructions les plus complexes.

# Pour s’assurer que les informations extraites sont pertinentes et ciblées, et que le modèle génère des résumés plus complets, j'ai fixé une taille maximale de tokens à 256.
# En fixant le paramètre 'temperature' à zéro, j'assure une coherance des résultats.

llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 256,
             api_key=os.getenv(OPENAI_API_KEY))

class TexteRecherche(BaseModel):
    texte: str

# Creation du premier endpoint pour l'extraction d'informations.

@app.post("/extraire-informations")
def extraire_informations(texte_recherche: TexteRecherche):

    # Détecter automatiquement des caractères speciaux indésirables
    
    original_text = texte_recherche.texte
    texte = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', original_text)
    texte = texte.encode('utf-8', 'ignore').decode('utf-8')

    try:
        # Utilisation de langchain pour extraire des informations.
        extraction_prompt = PromptTemplate(
        input_variables=["text_input"],
        template= """Extraits les informations spécifiques telles que les auteurs, les affiliations et les mots-clés du texte suivant: 
        \n {text_input}"""
        )
        
        extraction_chain = LLMChain(llm=llm, prompt=extraction_prompt)

        informations_extraites = extraction_chain.run(texte)

        return informations_extraites
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'extraction : {str(e)}")
        
# Creation du deuxieme endpoint pour le résumé du texte.

@app.post("/resumer-papier")
def resumer_papier(texte_recherche: TexteRecherche):

    # Détecter automatiquement des caractères speciaux indésirables
    
    original_text = texte_recherche.texte
    texte = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', original_text)
    texte = texte.encode('utf-8', 'ignore').decode('utf-8')
    
    try:
        # Utilisation de langchain pour générer un résumé
        
        resume_prompt = PromptTemplate(
        input_variables=["text_input"],
        template = """
        Fais un resumé du texte suivant:
        \n {text}
        """
        )
        
        resume_chain = LLMChain(llm=llm, prompt=resume_prompt)
        
        #resume_chain = load_summarize_chain(llm, 
                             #chain_type="map_reduce")
        
        resume_genere = resume_chain.run(texte)

        return {"resume": resume_genere}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de résumé : {str(e)}")



# La documentation automatique avec FastAPI est disponible sur http://127.0.0.1:8000/docs ou http://127.0.0.1:8000/redoc

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
    
