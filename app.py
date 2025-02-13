import streamlit as st
import torch
from transformers import pipeline

# Configuración del modelo
@st.cache_resource
def load_pipeline():
    model_id = "hSkilletGD/Llama-3.2-3B-trained"  # Ruta del modelo
    return pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

pipe = load_pipeline()

# Instrucción del asistente
instruction = """
Eres un asistente que ayuda a responder a los alumnos preguntas relacionadas exclusivamente con la Universidad Autónoma del Estado de México (UAEMex) Campus Ecatepec:
- La UAEMex es una universidad pública del Estado de México.
- Tu misión es proporcionar información clara y precisa sobre trámites, inscripciones, calendarios, becas, oferta académica, servicios escolares y otros temas relacionados con la UAEMex.
- Eres un chatbot amigable y amable tanto con alumnos como con profesores, sé respetuoso.
- Si la pregunta está fuera del contexto de la UAEMex o es sobre un tema no relacionado con la universidad, responde únicamente: 'Lo siento, solo puedo responder preguntas relacionadas con la Universidad Autónoma del Estado de México (UAEMex)'.
- Si la pregunta requiere información que no está disponible o que debe consultarse en la universidad directamente, responde solo 'Puedes consultar esta pregunta directamente en control escolar'.
- Si la pregunta es sobre pagos, colegiaturas o procesos administrativos específicos, sugiere contactar a servicios escolares.
"""

# Función para hacer la solicitud al modelo
def requestToLLM(req):
    outputs = pipe(
        [
            {"role": "system", "content": instruction},
            {"role": "user", "content": req},
        ],
        max_new_tokens=256,
        temperature=0.9
    )
    return outputs[0]["generated_text"][-1]["content"]

# Interfaz en Streamlit
st.title("Chatbot UAEMex")

user_input = st.text_input("Haz una pregunta sobre la UAEMex:")

if user_input:
    respuesta = requestToLLM(user_input)
    st.write("**Respuesta:**", respuesta)
