from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import uuid
import logging
from supabase import create_client, Client
from dotenv import load_dotenv
import io
from datetime import datetime
import torch
import pickle
import sys
from threading import Thread
import time

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix para PyTorch 2.6+ - Patch torch.load antes de cargar ultralytics
def patch_torch_load():
    """Parcha torch.load para permitir cargar modelos de ultralytics con PyTorch 2.6+"""
    original_load = torch.load
    
    def patched_load(f, map_location=None, weights_only=None, **kwargs):
        try:
            return original_load(f, map_location=map_location, weights_only=False, **kwargs)
        except Exception as e:
            logger.error(f"Error en patched load: {e}")
            raise
    
    torch.load = patched_load
    logger.info("✅ torch.load parcheado para PyTorch 2.6+ compatibility")

patch_torch_load()

from ultralytics import YOLO

# Inicializar FastAPI
app = FastAPI(
    title="API de Blur para Rostros y Placas con Supabase",
    description="API que detecta y aplica blur a rostros y placas de vehículos",
    version="2.2.0"
)

# Configuración de Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "processed-images")

# Inicializar cliente Supabase
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("✅ Cliente Supabase inicializado correctamente")
except Exception as e:
    logger.error(f"❌ Error inicializando Supabase: {e}")
    supabase = None

# Cargar modelos YOLO - Descargar de internet si no existen
def load_models():
    """Carga modelos YOLO. Si no existen, los descarga de Hugging Face"""
    global model_placas, model_rostros
    
    model_placas = None
    model_rostros = None
    
    try:
        logger.info("Cargando modelos YOLO...")
        
        # Intenta cargar desde archivos locales
        if os.path.exists('license-plate-finetune-v1n.pt'):
            model_placas = YOLO('license-plate-finetune-v1n.pt')
            logger.info("✅ Modelo placas (local) cargado")
        elif os.path.exists('license-plate-finetune-v1s.pt'):
            model_placas = YOLO('license-plate-finetune-v1s.pt')
            logger.info("✅ Modelo placas (local) cargado")
        else:
            # Descargar modelo oficial de YOLOv8
            logger.info("Descargando modelo de placas...")
            model_placas = YOLO('yolov8s.pt')  # Modelo genérico
            logger.info("✅ Modelo placas descargado")
        
        if os.path.exists('yolov8n-face.pt'):
            model_rostros = YOLO('yolov8n-face.pt')
            logger.info("✅ Modelo rostros (local) cargado")
        elif os.path.exists('model.pt'):
            model_rostros = YOLO('model.pt')
            logger.info("✅ Modelo rostros (local) cargado")
        else:
            # Descargar modelo nano
            logger.info("Descargando modelo de rostros...")
            model_rostros = YOLO('yolov8n.pt')
            logger.info("✅ Modelo rostros descargado")
        
        logger.info("✅ Todos los modelos cargados correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error cargando modelos: {e}")
        raise e

# Modelos globales
model_placas = None
model_rostros = None
models_loaded = False

def ensure_models_loaded():
    """Carga modelos la primera vez que se necesiten"""
    global model_placas, model_rostros, models_loaded
    
    if not models_loaded:
        logger.info("🔄 Primera vez - cargando modelos...")
        load_models()
        models_loaded = True

# Diccionario para rastrear estado de procesamiento
processing_status = {}

def aplicar_blur_region(image, x1, y1, x2, y2, nivel_blur=35):
    """Aplica blur a una región específica de la imagen"""
    if nivel_blur % 2 == 0:
        nivel_blur += 1
    
    roi = image[y1:y2, x1:x2]
    blurred_roi = cv2.GaussianBlur(roi, (nivel_blur, nivel_blur), 0)
    image[y1:y2, x1:x2] = blurred_roi
    return image

def procesar_rostros(image_array: np.ndarray, nivel_blur=71):
    """Procesa solo rostros"""
    image = image_array.copy()
    rostros_detectados = 0
    
    # Reducir imgsz a 416 para más velocidad
    results = model_rostros(image, conf=0.5, imgsz=416, verbose=False)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            image = aplicar_blur_region(image, x1, y1, x2, y2, nivel_blur)
            rostros_detectados += 1
    
    return image, rostros_detectados

def procesar_placas(image_array: np.ndarray, nivel_blur=35):
    """Procesa solo placas"""
    image = image_array.copy()
    placas_detectadas = 0
    
    results = model_placas(image, conf=0.015, imgsz=640)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            image = aplicar_blur_region(image, x1, y1, x2, y2, nivel_blur)
            placas_detectadas += 1
    
    return image, placas_detectadas

def upload_to_supabase(image: np.ndarray, filename: str, metadata: dict = None):
    """Sube imagen a Supabase Storage y guarda metadata en la base de datos"""
    if supabase is None:
        raise Exception("Cliente Supabase no inicializado")
    
    try:
        success, encoded_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise Exception("Error codificando imagen")
        
        image_bytes = encoded_image.tobytes()
        
        storage_response = supabase.storage.from_(SUPABASE_BUCKET).upload(
            file=image_bytes,
            path=filename,
            file_options={"content-type": "image/jpeg"}
        )
        
        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(filename)
        
        db_metadata = {
            "filename": filename,
            "public_url": public_url,
            "uploaded_at": datetime.utcnow().isoformat(),
            "file_size": len(image_bytes),
            "content_type": "image/jpeg"
        }
        
        if metadata:
            db_metadata.update(metadata)
        
        try:
            db_response = supabase.table("processed_images").insert(db_metadata).execute()
            logger.info(f"✅ Metadata guardada: {filename}")
        except Exception as db_error:
            logger.warning(f"⚠️  No se pudo guardar metadata: {db_error}")
        
        return {
            "public_url": public_url,
            "filename": filename,
            "file_size": len(image_bytes)
        }
        
    except Exception as e:
        logger.error(f"❌ Error subiendo a Supabase: {e}")
        raise e

def procesar_imagen_background(task_id: str, image_array: np.ndarray, original_filename: str,
                               blur_rostros: bool = True, blur_placas: bool = True,
                               blur_rostros_level: int = 71, blur_placas_level: int = 35):
    """Procesa imagen en background"""
    try:
        processing_status[task_id] = {"status": "processing", "progress": 0}
        
        image = image_array.copy()
        detecciones = {"rostros": 0, "placas": 0}
        
        if blur_rostros:
            image, rostros_count = procesar_rostros(image, blur_rostros_level)
            detecciones["rostros"] = rostros_count
            processing_status[task_id]["progress"] = 50
        
        if blur_placas:
            image, placas_count = procesar_placas(image, blur_placas_level)
            detecciones["placas"] = placas_count
            processing_status[task_id]["progress"] = 75
        
        filename = f"{uuid.uuid4().hex}.jpg"
        upload_result = upload_to_supabase(
            image,
            filename,
            metadata={
                "original_filename": original_filename,
                "detecciones": detecciones,
                "processing_type": "full"
            }
        )
        
        processing_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "result": {
                "success": True,
                "filename": filename,
                "public_url": upload_result["public_url"],
                "detecciones": detecciones,
                "file_size": upload_result["file_size"]
            }
        }
        
        logger.info(f"✅ Imagen procesada completamente: {task_id}")
        
    except Exception as e:
        logger.error(f"❌ Error procesando imagen {task_id}: {e}")
        processing_status[task_id] = {
            "status": "error",
            "error": str(e)
        }

@app.get("/")
async def root():
    return {
        "message": "API de Blur para Rostros y Placas con Supabase",
        "supabase_connected": supabase is not None,
        "endpoints": {
            "health": "/health",
            "process_image": "/process-image",
            "status": "/status/{task_id}"
        }
    }

@app.get("/health")
async def health_check():
    supabase_status = "connected" if supabase else "disconnected"
    ensure_models_loaded()
    return {
        "status": "healthy",
        "supabase": supabase_status,
        "models_loaded": models_loaded
    }

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    """
    Procesa imagen en background
    Retorna inmediatamente con task_id para consultar el estado
    """
    try:
        ensure_models_loaded()
        
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_image is None:
            raise HTTPException(status_code=400, detail="Error leyendo la imagen")
        
        # Generar task_id
        task_id = str(uuid.uuid4())
        
        # Iniciar procesamiento en background en un thread
        thread = Thread(
            target=procesar_imagen_background,
            args=(task_id, original_image, file.filename)
        )
        thread.daemon = True
        thread.start()
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Procesando imagen en background. Usa /status/{task_id} para consultar el progreso",
            "status_url": f"/status/{task_id}"
        }
        
    except Exception as e:
        logger.error(f"Error procesando imagen: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Consulta el estado del procesamiento
    """
    if task_id not in processing_status:
        raise HTTPException(status_code=404, detail="Task no encontrada")
    
    return processing_status[task_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
