from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import google.generativeai as genai
import os
from typing import Optional
import json # Importa la librería json
from dotenv import load_dotenv
from unidecode import unidecode
import pandas as pd
from azure.data.tables import TableClient
from azure.core import exceptions as azure_exceptions # Alias para evitar conflictos si hay otras excepciones 'exceptions'
from google.api_core import exceptions as google_exceptions
import asyncio
from rapidfuzz import fuzz
from fastapi.middleware.cors import CORSMiddleware


origins = [
    "http://localhost:8000",  # Ejemplo si tu frontend corre en localhost:8000
    "http://localhost:3000",  # Ejemplo si usas React en localhost:3000
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
    "null", # Permitir orígenes como file:// (para abrir el html directo), aunque no es seguro en prod.
    # Agrega aquí los dominios de tu frontend en producción
    "https://storagecategorization.z13.web.core.windows.net/"
]


# --- Configuración de Paths ---
# Obtiene la ruta del directorio donde está main.py
BASE_DIR = os.path.dirname(os.path.abspath(__name__))
# Construye la ruta completa al archivo de prompts
PROMPTS_FILE_PATH = os.path.join(BASE_DIR, 'data', 'prompts.json')

# Carga las variables de entorno desde un archivo .env
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# --- Cargar los prompts desde el archivo JSON al inicio ---
LOADED_PROMPTS: dict = {}
try:
    if os.path.exists(PROMPTS_FILE_PATH):
        with open(PROMPTS_FILE_PATH, 'r', encoding='utf-8') as f: # Usa encoding='utf-8'
            LOADED_PROMPTS = json.load(f)
        print(f"Prompts cargados desde {PROMPTS_FILE_PATH}")
    else:
        print(f"ADVERTENCIA: El archivo de prompts no se encontró en {PROMPTS_FILE_PATH}. Algunos endpoints podrían no funcionar correctamente.")
except json.JSONDecodeError:
    print(f"ERROR: El archivo {PROMPTS_FILE_PATH} no es un JSON válido.")
    # Puedes decidir salir o usar un diccionario vacío
    LOADED_PROMPTS = {}
except Exception as e:
    print(f"ERROR inesperado al cargar prompts desde {PROMPTS_FILE_PATH}: {e}")
    LOADED_PROMPTS = {}


# --- Configurar genai globalmente con la clave (si existe) ---
if GOOGLE_API_KEY:
     genai.configure(api_key=GOOGLE_API_KEY)
     print("API Key de Gemini cargada desde el entorno.")
else:
     print("ADVERTENCIA: GOOGLE_API_KEY no configurada. El endpoint /generate/ no funcionará.")

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
if not connection_string:
    print("Error: La variable de entorno 'AZURE_STORAGE_CONNECTION_STRING' no está configurada.")
    print("Por favor, configúrala antes de ejecutar el script.")
    exit()


# Define el modelo de datos para la solicitud entrante de Gemini (AHORA SIN LA CLAVE)
class GeminiRequest(BaseModel):
    # Puedes agregar un campo opcional para especificar qué prompt usar del archivo
    #prompt_key: Optional[str] = None # Clave para seleccionar un prompt del archivo
    user_description_input: str # La descripción del producto introducida para resumir


# Crea la instancia de la aplicación FastAPI
app = FastAPI(
    title="categorización" \
    " con Gemini" ,
    description="API para categorizar productos de consumo masivo bajo el arbol de categorías GPC de GS1 usando Gemini.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Lista de orígenes permitidos
    allow_credentials=True, # Permite cookies y credenciales (si las usas)
    allow_methods=["*"],    # Permite todos los métodos (GET, POST, PUT, DELETE, etc.)
    # O puedes especificar solo los que necesitas: allow_methods=["POST"]
    allow_headers=["*"],    # Permite todas las cabeceras (Content-Type, Authorization, etc.)
    # O puedes especificar solo las que necesitas: allow_headers=["Content-Type", "Accept"]
)

# --- Endpoint para Gemini (USA LA CLAVE GLOBAL Y PROMPTS CARGADOS) ---
@app.post("/summarize-description/")
async def generate_description_with_gemini(request: GeminiRequest):
    """
    Recibe entrada del usuario (descripción corta de un producto ej: Leche ALQUERIA uat semiscremada original familiar x1LTR)
     llama a la API de Gemini usando la clave del entorno y el prompt cargado.
    """
    # Verifica si la clave de la API está configurada
    if GOOGLE_API_KEY is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La API Key de Gemini no está configurada en el servidor."
        )

    # Verifica si la entrada del usuario está vacía
    if not request.user_description_input:
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La 'user_description_input' no puede estar vacía."
        )

    # --- Construye el prompt completo para Gemini ---
    final_prompt_text = ""
    # Si el cliente especificó una clave de prompt
    instruction = LOADED_PROMPTS.get('generate_product_description_prompt')    
    final_prompt_text = instruction.format(
        user_input=request.user_description_input
    )

    # Asegúrate de que el prompt final no esté vacío
    if not final_prompt_text.strip():
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El prompt final construido está vacío."
        )

    try:
        # Usa la clave configurada globalmente
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17') # Revisa el nombre del modelo si usas otro

        #print(f"Enviando prompt a Gemini:\n---\n{final_prompt_text}\n---") # Log para depuración
        response = model.generate_content(final_prompt_text)

        # Verifica si la respuesta tiene texto antes de acceder
        if response.text is None:
             raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="La API de Gemini no devolvió texto en la respuesta."
            )
        json_records = {"generated_description": response.text, "description": request.user_description_input}
        return json_records

    except google_exceptions as e:
        print(f"Error de API de Google: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error al interactuar con la API de Gemini: {e}"
        )
    except Exception as e:
        print(f"Error inesperado: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ocurrió un error interno: {e}"
        )

# --- NUEVO Endpoint: Generar descripción concisa y luego escoge el segmento de la GPC al que corresponde ---
@app.post("/generate-and-segment/")
async def generate_and_segmented_description(request: GeminiRequest):
    """
    Recibe descripción resumida, llama internamente a /generate/ para obtener texto de Gemini,
    y luego categoriza a nivel de segmento esa descripción.
    """
    # Llama a la función que maneja el endpoint /generate/.
    # Esto ejecuta la lógica de llamar a Gemini.
    # Captura cualquier HTTPException que pueda lanzar generate_text_with_gemini
    instruction = LOADED_PROMPTS.get('gpc_categorization_segment')
    final_prompt_text = instruction
        
    # Filtra por PartitionKey igual a 'CategoriaA'
    # La sintaxis del filtro usa OData
    table_name = "GPCsegments"
    try:
        table_client = TableClient.from_connection_string(
        conn_str=connection_string,
        table_name=table_name
    )
        try:
            list_of_entities = []
            all_entities = table_client.query_entities(query_filter=None)
            print(f"Entidades encontradas:")
            count = 0
            for entity in all_entities:
                list_of_entities.append(entity)
                #print(entity)
                count += 1
            print(f"Total de entidades encontradas: {count}")

        except Exception as e:
            print(f"Ocurrió un error al consultar por PartitionKey: {e}")
    except azure_exceptions.ClientAuthenticationError as e:
        print(f"Error de autenticación: Verifica tu cadena de conexión. {e}")
    except azure_exceptions.HttpResponseError as e:
        print(f"Error HTTP al conectar o validar la tabla: Status {e.status_code}, Mensaje: {e.message}")
    except Exception as e:
        print(f"Ocurrió un error inesperado al conectar o inicializar: {e}")



    df_segmentos = pd.DataFrame(list_of_entities)
    #df_segmentos = pd.read_csv('segmentos.csv', sep=';', encoding='latin-1')
    #df_segmentos['Segmento'] = df_segmentos['Segmento'].apply(unidecode)
    # Asegúrate de que el archivo CSV tiene una columna llamada 'segmento'
    # Define la lista de segmentos GPC
    segmentos = df_segmentos['Segmento'].tolist()


    # Asegúrate de que el prompt final no esté vacío
    if not final_prompt_text.strip():
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El prompt final construido está vacío."
        )
    
    
    try:
        gemini_response_dict = await generate_description_with_gemini(request)
        generated_text = gemini_response_dict.get("generated_description", "")
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

        #print(f"Enviando prompt a Gemini:\n---\n{final_prompt_text}\n---") # Log para depuración
        response = model.generate_content(final_prompt_text.format(
            "\n- ".join(segmentos),
            "\n- ".join(generated_text))
        )

        #print(response.text)
        condicion_booleana = df_segmentos['Segmento'] == response.text
        df_filtrado = df_segmentos[condicion_booleana]
        df_filtrado = df_filtrado.rename(columns={'RowKey': 'id_segmento'})
        df_filtrado['IA_description'] = generated_text
        df_filtrado['description'] = request.user_description_input
        df_filtrado.drop('PartitionKey', axis=1, inplace=True)
        json_records = df_filtrado.to_json(orient='records', indent=4)
        # --- Lógica de Interpretación/Procesamiento ---
        # Aquí es donde pondrías tu código para interpretar 'generated_text'
        
        # Puedes agregar más lógica aquí, como:
        # - Analizar sentimiento
        # - Extraer entidades clave
        # - Resumir (si el output de Gemini no era un resumen)
        # - Formatear el texto de otra manera

        # --- Fin Lógica de Interpretación/Procesamiento ---

        # Devuelve tanto el texto original generado como el interpretado
        #x = json.loads(json_records)
        #print(x[0])
        return json.loads(json_records)

    except HTTPException as e:
        # Si la llamada interna a generate_text_with_gemini lanzó un HTTPException,
        # lo relanzamos para que FastAPI lo maneje y devuelva el error al cliente.
        raise e
    except Exception as e:
        # Captura cualquier otro error que pudiera ocurrir *después* de la llamada a generate_text_with_gemini,
        # o si la llamada interna falló de forma inesperada sin un HTTPException
        print(f"Error durante la interpretación o llamada interna: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar la respuesta de Gemini: {e}"
        )


#--- NUEVO Endpoint: Generar descripción concisa y luego escoge la familia de la GPC a la que corresponde ---
@app.post("/generate-and-families/")
async def generate_description_and_segmented_family(request: GeminiRequest):
    """
    Recibe descripción resumida, llama internamente a /generate/ para obtener texto de Gemini,
    y luego categoriza a nivel de familia esa descripción.
    """
    # Llama a la función que maneja el endpoint /generate/.
    # Esto ejecuta la lógica de llamar a Gemini.
    # Captura cualquier HTTPException que pueda lanzar generate_text_with_gemini
    instruction = LOADED_PROMPTS.get('gpc_categorization_family')
    final_prompt_text = instruction
        
    # Filtra por PartitionKey igual a 'CategoriaA'
    # La sintaxis del filtro usa OData
    table_name = "GPCfamilies"
    MAX_RETRIES = 3       # Número máximo de intentos (1 intento inicial + MAX_RETRIES-1 reintentos)
    RETRY_DELAY_SECONDS = 2 # Tiempo de espera entre reintentos
    gemini_response_segment = None
    generated_segment_id = None
    success = False
    
    print(f"Intentando llamar a generate_and_segmented_description (hasta {MAX_RETRIES} intentos)...")

    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Intento {attempt + 1} de {MAX_RETRIES}...")
            # Llama a tu función asíncrona
            response_from_gemini = await generate_and_segmented_description(request)

            # --- Verificar el formato de la respuesta esperada ---
            # Asegúrate de que la respuesta sea una lista, no esté vacía,
            # que el primer elemento sea un diccionario, y que tenga la clave 'RowKey'.
            if (isinstance(response_from_gemini, list) and
                len(response_from_gemini) > 0 and
                isinstance(response_from_gemini[0], dict) and
                'id_segmento' in response_from_gemini[0]):

                # Si el formato es correcto, extrae el RowKey y marca como éxito
                gemini_response_segment = response_from_gemini # Guarda la respuesta completa si la necesitas
                generated_segment_id = response_from_gemini[0]['id_segmento']
                success = True
                print("  Llamada a Gemini exitosa y respuesta con formato esperado.")
                break # Sal del bucle de reintentos

            else:
                # Si el formato no es el esperado pero no hubo excepción
                print(f"  Intento {attempt + 1} fallido: Formato de respuesta inesperado de Gemini.")
                # Puedes imprimir la respuesta recibida para depurar
                # print(f"  Respuesta recibida: {response_from_gemini}")

        # --- Manejo de excepciones ---
        # Captura errores específicos de Google API si es posible
        except google_exceptions.GoogleAPIError as e:
            print(f"  Intento {attempt + 1} fallido: Error de Google API - {e}")
        # Captura errores generales de conexión o HTTP
        except (azure_exceptions.HttpResponseError, ConnectionError) as e:
            print(f"  Intento {attempt + 1} fallido: Error de conexión o HTTP - {e}")
        # Captura cualquier otra excepción inesperada
        except Exception as e:
            print(f"  Intento {attempt + 1} fallido: Error inesperado - {e}")

        # Si no fue el último intento, espera antes de reintentar
        if attempt < MAX_RETRIES - 1:
            print(f"  Esperando {RETRY_DELAY_SECONDS} segundos antes de reintentar...")
            await asyncio.sleep(RETRY_DELAY_SECONDS) # Usa asyncio.sleep en funciones async
        else:
            print(f"  Máximo número de reintentos ({MAX_RETRIES}) alcanzado.")


    # --- Procesar después de los reintentos ---
    if not success:
        # Si el bucle terminó sin éxito, lanza una excepción
        print("La llamada a generate_and_segmented_description falló después de varios intentos.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"No se pudo obtener una respuesta válida de Gemini después de {MAX_RETRIES} intentos."
    )

    # Si llegamos aquí, success es True y generated_segment_id tiene el valor
    print(f"El generated_segment_id obtenido es: {generated_segment_id}")

    
    
    try:
        list_of_entities = []
        filter_query_select = "id_segmento eq '{}'".format(generated_segment_id)
        table_client = TableClient.from_connection_string(
        conn_str=connection_string,
        table_name=table_name
    )
        try:
            all_entities = table_client.query_entities(query_filter=filter_query_select)
            print(f"Entidades encontradas:")
            count = 0
            for entity in all_entities:
                list_of_entities.append(entity)
                #print(entity)
                count += 1
            print(f"Total de entidades encontradas: {count}")

        except Exception as e:
            print(f"Ocurrió un error al consultar por PartitionKey: {e}")
    # except exceptions.ClientAuthenticationError as e:
    #     print(f"Error de autenticación: Verifica tu cadena de conexión. {e}")
    # except exceptions.HttpResponseError as e:
    #     print(f"Error HTTP al conectar o validar la tabla: Status {e.status_code}, Mensaje: {e.message}")
    except Exception as e:
        print(f"Ocurrió un error inesperado al conectar o inicializar: {e}")



    df_families = pd.DataFrame(list_of_entities)
    families = df_families['Familia'].tolist()
    #print(families.head())


    # Asegúrate de que el prompt final no esté vacío
    if not final_prompt_text.strip():
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El prompt final construido está vacío."
        )
    
    
    try:
        generated_text = gemini_response_segment[0]['description']
        print(generated_text)
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

        #print(f"Enviando prompt a Gemini:\n---\n{final_prompt_text}\n---") # Log para depuración
        response = model.generate_content(final_prompt_text.format(
            "\n- ".join(families),
            "\n- ".join(generated_text))
        )

        #--- Lógica de Interpretación/Procesamiento ---
        #Aquí es donde pondrías tu código para interpretar 'generated_text'
        
        #Puedes agregar más lógica aquí, como:
        #- Analizar sentimiento
        #- Extraer entidades clave
        #- Resumir (si el output de Gemini no era un resumen)
        #- Formatear el texto de otra manera

        #--- Fin Lógica de Interpretación/Procesamiento ---

        #Devuelve tanto el texto original generado como el interpretado
        gemini_response_segment[0]['Familia'] = response.text


        familia = gemini_response_segment[0]['Familia']
        filtro = df_families['Familia'] == familia
        df_filtrado = df_families[filtro]
        id_familia = df_filtrado.iloc[0]['RowKey']  
        gemini_response_segment[0]["id_familia"] = id_familia


        return gemini_response_segment[0]

    except HTTPException as e:
        #Si la llamada interna a generate_text_with_gemini lanzó un HTTPException,
        #lo relanzamos para que FastAPI lo maneje y devuelva el error al cliente.
        raise e
    except Exception as e:
        #Captura cualquier otro error que pudiera ocurrir *después* de la llamada a generate_text_with_gemini,
        #o si la llamada interna falló de forma inesperada sin un HTTPException
        print(f"Error durante la interpretación o llamada interna: {e}")
        raise HTTPException(
           status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
           detail=f"Error al procesar la respuesta de Gemini: {e}"
       )

#--- NUEVO Endpoint: Generar descripción concisa y luego escoge la familia de la GPC a la que corresponde ---
@app.post("/generate-and-classes/")
async def generate_description_and_segmented_classes(request: GeminiRequest):
    """
    Recibe descripción resumida, llama internamente a /generate/ para obtener texto de Gemini,
    y luego categoriza a nivel de familia esa descripción.
    """
    # Llama a la función que maneja el endpoint /generate/.
    # Esto ejecuta la lógica de llamar a Gemini.
    # Captura cualquier HTTPException que pueda lanzar generate_text_with_gemini
    instruction = LOADED_PROMPTS.get('gpc_categorization_clase')
    final_prompt_text = instruction
        
    # Filtra por PartitionKey igual a 'CategoriaA'
    # La sintaxis del filtro usa OData
    table_name = "GPCclasses"
    MAX_RETRIES = 3       # Número máximo de intentos (1 intento inicial + MAX_RETRIES-1 reintentos)
    RETRY_DELAY_SECONDS = 2 # Tiempo de espera entre reintentos
    response_from_gemini = None
    generated_segment_id = None
    success = False
    
    print(f"Intentando llamar a generate_and_segmented_description (hasta {MAX_RETRIES} intentos)...")

    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Intento {attempt + 1} de {MAX_RETRIES}...")
            # Llama a tu función asíncrona
            response_from_gemini = await generate_description_and_segmented_family(request)

            # --- Verificar el formato de la respuesta esperada ---
            # Asegúrate de que la respuesta sea una lista, no esté vacía,
            # que el primer elemento sea un diccionario, y que tenga la clave 'RowKey'.
            if (isinstance(response_from_gemini, dict) and
                'id_familia' in response_from_gemini):

                success = True
                print("  Llamada a Gemini exitosa y respuesta con formato esperado.")
                break # Sal del bucle de reintentos

            else:
                # Si el formato no es el esperado pero no hubo excepción
                print(f"  Intento {attempt + 1} fallido: Formato de respuesta inesperado de Gemini.")
                # Puedes imprimir la respuesta recibida para depurar
                # print(f"  Respuesta recibida: {response_from_gemini}")

        # --- Manejo de excepciones ---
        # Captura errores específicos de Google API si es posible
        except google_exceptions.GoogleAPIError as e:
            print(f"  Intento {attempt + 1} fallido: Error de Google API - {e}")
        # Captura errores generales de conexión o HTTP
        except (azure_exceptions.HttpResponseError, ConnectionError) as e:
            print(f"  Intento {attempt + 1} fallido: Error de conexión o HTTP - {e}")
        # Captura cualquier otra excepción inesperada
        except Exception as e:
            print(f"  Intento {attempt + 1} fallido: Error inesperado - {e}")

        # Si no fue el último intento, espera antes de reintentar
        if attempt < MAX_RETRIES - 1:
            print(f"  Esperando {RETRY_DELAY_SECONDS} segundos antes de reintentar...")
            await asyncio.sleep(RETRY_DELAY_SECONDS) # Usa asyncio.sleep en funciones async
        else:
            print(f"  Máximo número de reintentos ({MAX_RETRIES}) alcanzado.")


    # --- Procesar después de los reintentos ---
    if not success:
        # Si el bucle terminó sin éxito, lanza una excepción
        print("La llamada a generate_and_segmented_description falló después de varios intentos.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"No se pudo obtener una respuesta válida de Gemini después de {MAX_RETRIES} intentos."
    )



    
    try:
        list_of_entities = []
        generated_family_id = response_from_gemini['id_familia']
        filter_query_select = "id_familia eq '{}'".format(generated_family_id)
        table_client = TableClient.from_connection_string(
        conn_str=connection_string,
        table_name=table_name
    )    
        try:
            entities = table_client.query_entities(query_filter=filter_query_select)
            print(f"Entidades encontradas:")
            count = 0
            for entity in entities:
                list_of_entities.append(entity)
                #print(entity)
                count += 1
            print(f"Total de entidades encontradas: {count}")
            df_clases = pd.DataFrame(list_of_entities)
            #print(df_clases)
            clases = df_clases['Clase'].tolist()

        except Exception as e:
            print(f"Ocurrió un error al consultar por PartitionKey: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado al conectar o inicializar: {e}")


        
    try:
        generated_text = response_from_gemini['IA_description']
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

        #print(f"Enviando prompt a Gemini:\n---\n{final_prompt_text}\n---") # Log para depuración
        response = model.generate_content(final_prompt_text.format(
            "\n- ".join(clases),
            "\n- ".join(generated_text))
        )
        
        #Devuelve tanto el texto original generado como el interpretado
        partes = response.text.split(':', 1)
        if len(partes) > 1:
            # Si hay dos partes, toma la segunda
            clase = partes[1].lstrip()
        else:
            clase = str(partes[0])
        response_from_gemini['clase'] = clase


        clase = response_from_gemini['clase']
        df_clases['Similitud'] = df_clases['Clase'].apply(lambda x: fuzz.ratio(clase, x))
        df_fila_mas_similar = df_clases.nlargest(1, 'Similitud')
        id_clase = df_fila_mas_similar.iloc[0]['RowKey']
        response_from_gemini["id_clase"] = id_clase

        return response_from_gemini

    except HTTPException as e:
        #Si la llamada interna a generate_text_with_gemini lanzó un HTTPException,
        #lo relanzamos para que FastAPI lo maneje y devuelva el error al cliente.
        raise e
    except Exception as e:
        #Captura cualquier otro error que pudiera ocurrir *después* de la llamada a generate_text_with_gemini,
        #o si la llamada interna falló de forma inesperada sin un HTTPException
        print(f"Error durante la interpretación o llamada interna: {e}")
        raise HTTPException(
           status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
           detail=f"Error al procesar la respuesta de Gemini: {e}"
       )


# Endpoint raíz
@app.get("/")
async def read_root():
    status_msg = "configurada" if GOOGLE_API_KEY else "NO configurada"
    prompts_status = f"{len(LOADED_PROMPTS)} cargados" if LOADED_PROMPTS else "NO cargados o error"
    return {
        "message": "API lista.",
        "gemini_key_status": status_msg,
        "prompts_status": prompts_status,
        "endpoints": {
            "/generate/ (POST)": "Requiere user_input y opcionalmente prompt_key. Usa la clave del entorno y prompts cargados."
        }
    }