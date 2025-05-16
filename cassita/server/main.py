from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
import io
import time
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import logging

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация
MODEL_NAME = "hafsa000/interior-design"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_STEPS = 25  # Оптимальное количество шагов для CPU

pipe = None

@app.on_event("startup")
async def load_model():
    global pipe
    try:
        logger.info(f"Loading model {MODEL_NAME} on {DEVICE}...")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=TORCH_DTYPE,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        if DEVICE == "cuda":
            pipe = pipe.to(DEVICE)
            pipe.enable_attention_slicing()
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except:
                logger.warning("xformers not available, using default attention")
        
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise

@app.post("/generate")
async def generate_design(
    prompt: str = Form(...),
    strength: float = Form(0.7),
    guidance_scale: float = Form(7.5),
    steps: int = Form(MAX_STEPS),
    image: UploadFile = File(None)
):
    if not pipe:
        raise HTTPException(500, detail="Model not loaded")
    
    try:
        logger.info(f"Starting generation for prompt: {prompt}")
        start_time = time.time()
        
        # Обработка изображения
        init_image = None
        if image and image.filename:
            try:
                image_data = await image.read()
                init_image = Image.open(io.BytesIO(image_data)).convert("RGB")
                init_image = init_image.resize((512, 512))
            except Exception as e:
                raise HTTPException(400, detail=f"Invalid image: {str(e)}")

        steps = min(steps, MAX_STEPS)
        
        # Генерация
        generator = torch.Generator(device=DEVICE).manual_seed(42)
        
        try:
            if init_image:
                result = pipe(
                    prompt=prompt,
                    image=init_image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps,
                    generator=generator
                )
            else:
                result = pipe(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps,
                    generator=generator
                )
        except torch.cuda.OutOfMemoryError:
            raise HTTPException(500, detail="CUDA out of memory. Try smaller image or fewer steps.")
        
        # Конвертация в PNG
        img_byte_arr = io.BytesIO()
        result.images[0].save(img_byte_arr, format="PNG")
        image_bytes = img_byte_arr.getvalue()
        
        logger.info(f"Generation completed in {time.time() - start_time:.2f}s")
        
        # Возвращаем бинарные данные изображения
        return Response(
            content=image_bytes,
            media_type="image/png",
            headers={
                "X-Time-Taken": str(time.time() - start_time),
                "X-Steps-Used": str(steps)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(500, detail=str(e))