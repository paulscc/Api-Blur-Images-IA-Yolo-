from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import os
import uuid
import logging
from supabase import create_client, Client
from dotenv import load_dotenv
import io
from datetime import datetime

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="API de Blur para Rostros y Placas con Supabase",
    description="API que detecta y aplica blur a rostros y placas de vehículos y guarda en Supabase",
    version="2.1.0"
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

# Cargar modelos YOLO
try:
    logger.info("Cargando modelos YOLO...")
    model_placas = YOLO('license-plate-finetune-v1s.pt')
    model_rostros = YOLO('model.pt')
    logger.info("✅ Modelos YOLO cargados correctamente")
except Exception as e:
    logger.error(f"❌ Error cargando modelos: {e}")
    raise e

def aplicar_blur_region(image, x1, y1, x2, y2, nivel_blur=35):
    """Aplica blur a una región específica de la imagen"""
    if nivel_blur % 2 == 0:
        nivel_blur += 1
    
    roi = image[y1:y2, x1:x2]
    blurred_roi = cv2.GaussianBlur(roi, (nivel_blur, nivel_blur), 0)
    image[y1:y2, x1:x2] = blurred_roi
    return image

def procesar_rostros(image_array: np.ndarray, nivel_blur=71):
    """Procesa solo rostros con tu modelo específico"""
    image = image_array.copy()
    rostros_detectados = 0
    
    results = model_rostros(image, conf=0.5)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            logger.info(f"Rostro detectado - Clase: {cls}, Confianza: {conf:.3f}")
            
            image = aplicar_blur_region(image, x1, y1, x2, y2, nivel_blur)
            rostros_detectados += 1
    
    return image, rostros_detectados

def procesar_placas(image_array: np.ndarray, nivel_blur=35):
    """Procesa solo placas con confianza baja como en tu código"""
    image = image_array.copy()
    placas_detectadas = 0
    
    results = model_placas(image, conf=0.015)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            logger.info(f"Placa detectada - Confianza: {conf:.3f}")
            
            image = aplicar_blur_region(image, x1, y1, x2, y2, nivel_blur)
            placas_detectadas += 1
    
    return image, placas_detectadas

def upload_to_supabase(image: np.ndarray, filename: str, metadata: dict = None):
    """
    Sube imagen a Supabase Storage y guarda metadata en la base de datos
    """
    if supabase is None:
        raise Exception("Cliente Supabase no inicializado")
    
    try:
        # Convertir imagen a bytes
        success, encoded_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise Exception("Error codificando imagen")
        
        image_bytes = encoded_image.tobytes()
        
        # Subir a Supabase Storage
        storage_response = supabase.storage.from_(SUPABASE_BUCKET).upload(
            file=image_bytes,
            path=filename,
            file_options={"content-type": "image/jpeg"}
        )
        
        # Obtener URL pública
        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(filename)
        
        # Preparar metadata para la base de datos
        db_metadata = {
            "filename": filename,
            "public_url": public_url,
            "uploaded_at": datetime.utcnow().isoformat(),
            "file_size": len(image_bytes),
            "content_type": "image/jpeg"
        }
        
        # Agregar metadata adicional si se proporciona
        if metadata:
            db_metadata.update(metadata)
        
        # Guardar en la tabla 'processed_images' (debes crearla en Supabase)
        try:
            db_response = supabase.table("processed_images").insert(db_metadata).execute()
            logger.info(f"✅ Metadata guardada en base de datos: {db_metadata['filename']}")
        except Exception as db_error:
            logger.warning(f"⚠️  No se pudo guardar metadata en BD: {db_error}")
        
        return {
            "public_url": public_url,
            "filename": filename,
            "file_size": len(image_bytes),
            "storage_response": storage_response
        }
        
    except Exception as e:
        logger.error(f"❌ Error subiendo a Supabase: {e}")
        raise e

def procesar_imagen_completa(image_array: np.ndarray, blur_rostros: bool = True, blur_placas: bool = True, 
                            blur_rostros_level: int = 71, blur_placas_level: int = 35):
    """
    Procesa imagen con ambos modelos
    """
    image = image_array.copy()
    detecciones = {"rostros": 0, "placas": 0}
    
    # Procesar rostros primero
    if blur_rostros:
        try:
            image, rostros_count = procesar_rostros(image, blur_rostros_level)
            detecciones["rostros"] = rostros_count
        except Exception as e:
            logger.error(f"Error procesando rostros: {e}")
    
    # Luego procesar placas
    if blur_placas:
        try:
            image, placas_count = procesar_placas(image, blur_placas_level)
            detecciones["placas"] = placas_count
        except Exception as e:
            logger.error(f"Error procesando placas: {e}")
    
    return image, detecciones

@app.get("/")
async def root():
    return {
        "message": "API de Blur para Rostros y Placas con Supabase",
        "supabase_connected": supabase is not None,
        "endpoints": {
            "health": "/health",
            "process_image": "/process-image",
            "process_with_options": "/process-image-options",
            "process_faces_only": "/process-faces",
            "process_plates_only": "/process-plates"
        }
    }

@app.get("/health")
async def health_check():
    supabase_status = "connected" if supabase else "disconnected"
    return {
        "status": "healthy",
        "supabase": supabase_status,
        "models_loaded": True
    }

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    """
    Endpoint simple - procesa imagen con ambos modelos y sube a Supabase
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        # Leer imagen
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_image is None:
            raise HTTPException(status_code=400, detail="Error leyendo la imagen")
        
        # Procesar imagen
        imagen_procesada, detecciones = procesar_imagen_completa(original_image)
        
        # Generar nombre único
        filename = f"{uuid.uuid4().hex}.jpg"
        
        # Subir a Supabase
        upload_result = upload_to_supabase(
            imagen_procesada, 
            filename,
            metadata={
                "original_filename": file.filename,
                "detecciones": detecciones,
                "processing_type": "full"
            }
        )
        
        logger.info(f"✅ Imagen procesada y subida: {detecciones}")
        
        return {
            "success": True,
            "filename": filename,
            "public_url": upload_result["public_url"],
            "detecciones": detecciones,
            "file_size": upload_result["file_size"]
        }
        
    except Exception as e:
        logger.error(f"Error procesando imagen: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

@app.post("/process-image-options")
async def process_image_with_options(
    file: UploadFile = File(...),
    blur_faces: bool = True,
    blur_plates: bool = True,
    face_blur_level: int = 71,
    plate_blur_level: int = 35
):
    """
    Endpoint avanzado con opciones personalizadas
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_image is None:
            raise HTTPException(status_code=400, detail="Error leyendo la imagen")
        
        # Procesar con opciones
        imagen_procesada, detecciones = procesar_imagen_completa(
            original_image, blur_faces, blur_plates, face_blur_level, plate_blur_level
        )
        
        filename = f"{uuid.uuid4().hex}.jpg"
        
        upload_result = upload_to_supabase(
            imagen_procesada,
            filename,
            metadata={
                "original_filename": file.filename,
                "detecciones": detecciones,
                "processing_type": "custom",
                "options_used": {
                    "blur_faces": blur_faces,
                    "blur_plates": blur_plates,
                    "face_blur_level": face_blur_level,
                    "plate_blur_level": plate_blur_level
                }
            }
        )
        
        return {
            "success": True,
            "filename": filename,
            "public_url": upload_result["public_url"],
            "detecciones": detecciones,
            "opciones_usadas": {
                "blur_faces": blur_faces,
                "blur_plates": blur_plates,
                "face_blur_level": face_blur_level,
                "plate_blur_level": plate_blur_level
            }
        }
        
    except Exception as e:
        logger.error(f"Error procesando imagen: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

@app.post("/process-faces")
async def process_faces_only(
    file: UploadFile = File(...),
    blur_level: int = 71
):
    """
    Procesa solo rostros
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_image is None:
            raise HTTPException(status_code=400, detail="Error leyendo la imagen")
        
        # Procesar solo rostros
        imagen_procesada, rostros_count = procesar_rostros(original_image, blur_level)
        
        filename = f"faces_{uuid.uuid4().hex}.jpg"
        
        upload_result = upload_to_supabase(
            imagen_procesada,
            filename,
            metadata={
                "original_filename": file.filename,
                "rostros_detectados": rostros_count,
                "blur_level": blur_level,
                "processing_type": "faces_only"
            }
        )
        
        return {
            "success": True,
            "filename": filename,
            "public_url": upload_result["public_url"],
            "rostros_detectados": rostros_count,
            "blur_level": blur_level
        }
        
    except Exception as e:
        logger.error(f"Error procesando rostros: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando rostros: {str(e)}")

@app.post("/process-plates")
async def process_plates_only(
    file: UploadFile = File(...),
    blur_level: int = 35
):
    """
    Procesa solo placas
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_image is None:
            raise HTTPException(status_code=400, detail="Error leyendo la imagen")
        
        # Procesar solo placas
        imagen_procesada, placas_count = procesar_placas(original_image, blur_level)
        
        filename = f"plates_{uuid.uuid4().hex}.jpg"
        
        upload_result = upload_to_supabase(
            imagen_procesada,
            filename,
            metadata={
                "original_filename": file.filename,
                "placas_detectadas": placas_count,
                "blur_level": blur_level,
                "processing_type": "plates_only"
            }
        )
        
        return {
            "success": True,
            "filename": filename,
            "public_url": upload_result["public_url"],
            "placas_detectadas": placas_count,
            "blur_level": blur_level
        }
        
    except Exception as e:
        logger.error(f"Error procesando placas: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando placas: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
