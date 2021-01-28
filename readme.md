# Facebook Messenger Bot

Chatbot generativo para Facebook que responde usando un modelo seq2seq.
- [Base para el chatbot](https://github.com/santirom/chatbot-bancolombia2)
- [Base para el modelo](https://towardsdatascience.com/generative-chatbots-using-the-seq2seq-model-d411c8738ab5)

## Pasos antes de desplegar
1. Cuando se agreguen más datos en el dataset se debe generar de nuevo el archivo con los datos pre-procesados:
    
    `python pre_process_data.py`

## Pasos para correr localmente el chatbot:
1. Crear un ambiente virtual.
2. Instalar los requerimientos:
   
    `pip install requirements.txt`
   
3. Correr el código:
    
    `python app.py`