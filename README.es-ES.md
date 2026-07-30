

# Sistema de Detección de Anomalías en Video

Sistema impulsado por IA para detectar eventos inusuales en videos de vigilancia. Sube un video y el sistema identificará automáticamente los fotogramas donde ocurre algo anormal.

**Rendimiento:** 92,47% de precisión, 0,7438 de AUC en el conjunto de datos UCSD Ped2

> **⚠️ Importante**: Este modelo está entrenado con grabaciones de vigilancia peatonal al aire libre. Los mejores resultados se obtienen en entornos similares. Para otros tipos de cámaras, el modelo puede necesitar reentrenarse con tus grabaciones específicas.

---

## 🎯 ¿Qué hace esto?

Este sistema analiza videos de vigilancia fotograma por fotograma para detectar anomalías: eventos inusuales que difieren de los patrones normales. Los ejemplos incluyen:

- **Patrones de movimiento inusuales** (correr, comportamiento errático)
- **Objetos inesperados** (vehículos donde no deberían estar, objetos abandonados)
- **Densidad de multitud anormal** (reuniones repentinas o espacios vacíos)
- **Actividades irregulares** (personas en áreas restringidas, gestos inusuales)

**¿Cómo funciona?:** El modelo de IA aprende cómo se ve lo "normal" a partir de los datos de entrenamiento. Cuando ve algo diferente, lo marca como una anomalía basándose en qué tan mal puede reconstruir el fotograma inusual.

---

## 🚀 Inicio rápido

### Usa la demostración en vivo (sin instalación)

**La forma más fácil de probar el sistema:**

1. **Visita:** https://video-anomaly-detection-dashboard.onrender.com
2. **Sube** un video de vigilancia (MP4, AVI, MOV)
3. **Visualiza los resultados** con una línea de tiempo interactiva y visor de fotogramas

**Para desarrolladores:** Documentación de la API en https://video-anomaly-detection-api.onrender.com/docs

---

### Ejecutar localmente

**Requisitos:**
- Python 3.10+
- 2 GB de espacio en disco
- Opcional: GPU NVIDIA para un procesamiento más rápido

**Configuración:**

```bash
# Install dependencies
pip install -r requirements.txt

# Start API backend (Terminal 1)
python app.py
# API available at http://localhost:8000

# Launch dashboard (Terminal 2)
streamlit run dashboard.py
# Dashboard opens at http://localhost:8501
```

**¿Por qué ejecutarlo localmente?**
- **Procesamiento más rápido** con GPU (0,2s frente a 5-10s por video)
- **Uso sin conexión**: no se requiere internet
- **Privacidad**: tus videos nunca salen de tu computadora
- **Configuración personalizada**: ajusta todos los parámetros

---

## 📊 Características

### Panel interactivo
- **Carga de videos arrastrando y soltando**
- **Línea de tiempo interactiva** que muestra los errores de reconstrucción
- **Ajuste de umbral en tiempo real**: cambia la sensibilidad sin reprocesar
- **Visor de fotogramas**: inspecciona anomalías específicas
- **Exportar resultados** a JSON o CSV para informes

### API REST
- **Solicitud POST simple** para análisis de video
- **Respuesta JSON** con puntuaciones de anomalía por fotograma
- **Umbrales ajustables** mediante endpoints de API
- **Documentación Swagger** en `/docs`

### Configuraciones preestablecidas de umbral

Ajusta la sensibilidad según tus necesidades:

| Preajuste | Tasa de anomalía | Ideal para |
|--------|--------------|----------|
| **Conservador** | 5% | Minimizar falsas alarmas |
| **Equilibrado** | 10% | Vigilancia general (predeterminado) |
| **Moderado** | 25% | Monitoreo de alta sensibilidad |
| **Sensible** | 40% | Detección máxima (más alertas) |

---

## 💡 Casos de uso

### Monitoreo de seguridad
```python
# Analyze camera feed for unusual activity
response = requests.post(
    "https://video-anomaly-detection-api.onrender.com/analyze-video",
    files={"file": open("camera_feed.mp4", "rb")}
)

if response.json()["anomaly_rate"] > 0.15:
    send_security_alert()  # Trigger alert if >15% anomalous frames
```

### Análisis minorista
- Detectar comportamientos inusuales de los clientes
- Identificar patrones potenciales de hurto en tiendas
- Monitorear el acceso a áreas restringidas

### Seguridad pública
- Identificar anomalías en la multitud
- Detectar objetos abandonados
- Monitorear irregularidades en el flujo peatonal

### Control de calidad
- Detección de anomalías en líneas de fabricación
- Monitoreo de procesos
- Detección de fallas en equipos

---

## 🎛️ Referencia de la API

### Analizar video

```http
POST /analyze-video
Content-Type: multipart/form-data
```

**Ejemplo:**
```bash
curl -X POST "https://video-anomaly-detection-api.onrender.com/analyze-video" \
  -F "file=@your_video.mp4"
```

**Respuesta:**
```json
{
  "frame_count": 60,
  "anomaly_count": 8,
  "anomaly_rate": 0.13,
  "anomaly_scores": [0.002, 0.008, 0.012, ...],
  "processing_time": 0.85,
  "model_info": {
    "device": "cuda",
    "threshold": 0.005069
  }
}
```

### Establecer preajuste de umbral

```http
POST /set-threshold-preset
Content-Type: application/json

{
  "preset": "balanced"  // conservative, balanced, moderate, sensitive
}
```

### Calibrar umbral

```http
POST /calibrate-threshold
Content-Type: application/json

{
  "target_anomaly_rate": 0.10  // Target 10% anomaly rate
}
```

### Métricas de Prometheus (Monitoreo de producción)

Para monitoreo de DevOps/infraestructura, la API expone métricas compatibles con Prometheus:

```http
GET /metrics/prometheus
```

**Métricas expuestas:**
- `anomaly_detection_requests_total` - Conteo de solicitudes por endpoint y estado
- `anomaly_detection_request_latency_seconds` - Histograma de latencia de solicitudes
- `anomaly_detection_frames_processed_total` - Total de fotogramas procesados
- `anomaly_detection_anomalies_total` - Total de anomalías detectadas
- `anomaly_detection_active_jobs` - Trabajos en segundo plano activos
- `anomaly_detection_gpu_memory_bytes` - Uso de memoria GPU (si está disponible)
- `anomaly_detection_inference_latency_seconds` - Latencia de inferencia del modelo por lote

**Uso con Prometheus:**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'anomaly-detection'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics/prometheus'
```

**Documentación completa de la API:** Visita el endpoint `/docs` para la interfaz Swagger interactiva

---

---

## 🏗️ Arquitectura del sistema

El sistema utiliza un **autoencoder convolucional**, una red neuronal entrenada para reconstruir grabaciones normales de vigilancia. Así es como detecta las anomalías:

```
Video Upload
    ↓
Frame Extraction (OpenCV)
    ↓
Preprocessing (Grayscale, 64×64 resize)
    ↓
AI Model (Autoencoder)
    ↓
Reconstruction Error Calculation
    ↓
Threshold Comparison
    ↓
Anomaly Flags + Scores
```

**Concepto clave:** El modelo aprende a recrear con precisión los fotogramas "normales". Cuando se encuentra con algo inusual, la calidad de la reconstrucción disminuye; este pico de error indica una anomalía.

**Detalles técnicos:**
- **Entrada:** Fotogramas en escala de grises de 64×64
- **Arquitectura:** Codificador (comprimir) → Espacio latente (256 dimensiones) → Decodificador (reconstruir)
- **Salida:** Error de reconstrucción por fotograma (escala de 0.0 a 1.0)
- **Umbral:** Corte estadístico (típicamente el percentil 95 de los errores del conjunto de validación)

---

## 📂 Estructura del proyecto

```
├── app.py                    # FastAPI web service
├── dashboard.py              # Streamlit interactive UI
├── settings.py               # Configuration management
├── models/
│   ├── autoencoder.py        # Neural network architecture
│   └── detector.py           # Training and inference
├── data/
│   ├── preprocessing.py      # Video frame extraction
│   └── dataset.py            # Data loading utilities
├── outputs/
│   └── trained_model.pth     # Pre-trained model weights
└── requirements.txt          # Python dependencies
```

---

## 🔧 Configuración

La configuración predeterminada funciona para la mayoría de los casos. Personalízala mediante variables de entorno o un archivo `.env`:

```bash
# File size limits
APP_MAX_FILE_SIZE_MB=100              # Max video file size
APP_MAX_VIDEO_DURATION_SEC=300        # Max 5 minutes

# Processing
APP_BATCH_SIZE=64                     # Frames processed per batch
APP_DEVICE=cuda                       # Use 'cpu' to force CPU processing

# Thresholds
APP_THRESHOLD=0.005069                # Anomaly detection threshold
```

**Cuándo ajustar:**
- **Videos grandes:** Reduce `APP_BATCH_SIZE` si se agota la memoria
- **Sin GPU:** Establece `APP_DEVICE=cpu` (espera un procesamiento más lento)
- **Demasiadas alertas:** Aumenta el valor de `APP_THRESHOLD`
- **Anomalías no detectadas:** Disminuye el valor de `APP_THRESHOLD`

---

## 🎓 Rendimiento del modelo

**Conjunto de datos de entrenamiento:** UCSD Ped2 (vigilancia peatonal al aire libre)

**Métricas:**
- **Precisión:** 92,47% - Cuando el sistema marca una anomalía, generalmente es correcta
- **Sensibilidad (Recall):** 83,78% - Detecta la mayoría de las anomalías reales
- **Puntuación F1:** 87,91% - Rendimiento equilibrado
- **AUC:** 0,7438 - Buena discriminación entre lo normal y lo anormal

**Qué significa esto:**
- **Bajos falsos positivos:** Alertas confiables
- **Buena detección:** Captura la mayoría de los eventos inusuales
- **Ideal para:** Vigilancia general, detección de actividades inusuales
- **Limitaciones:** El rendimiento disminuye con grabaciones muy diferentes a los datos de entrenamiento

---

## 🛠️ Uso avanzado

### Exportación ONNX (Opcional: solo para implementaciones avanzadas)

**¿Qué es ONNX?** Un formato de modelo multiplataforma para implementaciones especializadas.

**Cuándo usarlo:**
- Implementación en dispositivos periféricos (Raspberry Pi, Jetson Nano)
- Plataformas que requieren ONNX (Azure ML, AWS SageMaker)
- Optimizaciones específicas de hardware (TensorRT para NVIDIA, OpenVINO para Intel)

**Cuándo NO usarlo:**
- Implementaciones regulares (el modelo PyTorch ya es rápido)
- Hospedaje en la nube (Render, AWS Lambda) - PyTorch funciona bien
- Uso local - sin beneficio adicional

**Importante:** La exportación ONNX NO mejora la precisión (es el mismo modelo en un formato diferente). La mejora de velocidad solo ocurre con aceleradores de hardware especializados.

**Comando de exportación:**
```bash
# Basic export
python export_model.py --output outputs/model.onnx

# With optimizations and validation
python export_model.py --output outputs/model.onnx --optimize --validate --benchmark
```

**Usar el modelo ONNX:**
```python
import onnxruntime as ort
session = ort.InferenceSession("outputs/model.onnx")
output = session.run(None, {"input": preprocessed_frames})
```

### Reentrenamiento con tus propios datos

**¿Por qué reentrenar?**
- El modelo actual está entrenado con grabaciones peatonales al aire libre (UCSD Ped2)
- Tus cámaras pueden estar en interiores, comercios, estacionamientos, etc.
- Reentrenar con tus propias grabaciones mejora la precisión para tu entorno específico

**Paso 1: Obtener datos de entrenamiento**

**Opción A: Usar el conjunto de datos UCSD Ped2 (Datos de entrenamiento originales)**
```bash
# Download from official source
# Visit: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
# Download: UCSD Anomaly Detection Dataset - Ped2

# Extract to project directory
# Expected structure:
# data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/
# data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/
```

**Opción B: Usar tus propias grabaciones de cámara**
```bash
# Create data directory
mkdir -p data/my_cameras/normal_behavior/

# Add your videos (normal behavior only, no anomalies)
# - At least 10-20 videos, 30-60 seconds each
# - Typical daily operations, normal foot traffic
# - Consistent lighting and camera angles
# - MP4, AVI, or MOV format

# Example structure:
# data/my_cameras/normal_behavior/
#   ├── camera1_morning_20250113.mp4
#   ├── camera1_afternoon_20250113.mp4
#   ├── camera2_evening_20250113.mp4
#   └── ...
```

**Paso 2: Entrenar el modelo**

**Usando UCSD Ped2 (Conjunto de datos original):**
```bash
python main.py --mode ucsd --dataset_name ped2 \
    --data_path data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/ \
    --epochs 50
```

**Usando tus propias grabaciones:**
```bash
python main.py --mode custom \
    --data_path data/my_cameras/normal_behavior/ \
    --epochs 50 \
    --batch_size 64
```

**Salida del entrenamiento:**
```
Epoch 1/50: Loss=0.0234 (2m 15s)
Epoch 2/50: Loss=0.0187 (2m 12s)
...
✓ Training complete!
✓ Model saved to: outputs/trained_model.pth
✓ Threshold calibrated: 0.005234
```

**Paso 3: Probar el nuevo modelo**

```bash
# Restart API to load new model
python app.py

# Test with your videos via dashboard
streamlit run dashboard.py
```

**Consejos de entrenamiento:**
- **Más datos = mejor precisión** (apunta a 30+ minutos de grabación)
- **Condiciones consistentes:** Iluminación, clima y hora del día similares
- **Solo comportamiento normal:** No incluyas anomalías en los datos de entrenamiento
- **Se recomienda GPU:** El entrenamiento toma 10-30 minutos con GPU frente a 2-4 horas en CPU
- **Monitorea la pérdida:** Debe disminuir constantemente; si se estanca temprano, añade más datos

### Prueba rápida (datos sintéticos)

¿No tienes grabaciones reales aún? Genera videos de prueba:

```bash
python create_realistic_test_videos.py
# Creates 5 test videos in test_videos/
# Mix of normal pedestrian motion + anomalies

# Analyze them
streamlit run dashboard.py
# Upload videos from test_videos/
```

### Procesamiento por lotes

Procesa múltiples videos de forma programática:

```python
import requests
import os

api_url = "https://video-anomaly-detection-api.onrender.com/analyze-video"

video_dir = "surveillance_footage/"
for filename in os.listdir(video_dir):
    if filename.endswith((".mp4", ".avi", ".mov")):
        with open(os.path.join(video_dir, filename), "rb") as video:
            response = requests.post(api_url, files={"file": video})
            result = response.json()
            
            # Log high-anomaly videos
            if result["anomaly_rate"] > 0.20:
                print(f"⚠️  {filename}: {result['anomaly_count']} anomalies")
```

### Implementación con Docker

Ejecuta el sistema en un contenedor:

```bash
# Build image
docker build -t anomaly-detector .

# Run container
docker run -p 8000:8000 anomaly-detector

# API available at http://localhost:8000
```

---

## 📚 Documentación

- **[DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md)** - Guía completa de características del panel
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Instrucciones de configuración e implementación
- **[deployment/README.md](deployment/README.md)** - Información sobre los servicios en vivo
- **Docs API:** Visita `/docs` en cualquier instancia de API en ejecución

---

## ❓ Solución de problemas

**Alta tasa de falsos positivos:**
- Aumenta el umbral usando el control deslizante del panel o la API
- Cambia al preajuste "Conservador"
- Considera reentrenar con tus grabaciones específicas

**Anomalías evidentes no detectadas:**
- Disminuye el umbral usando el control deslizante del panel
- Cambia al preajuste "Sensible"
- Verifica que el tipo de anomalía coincida con los datos de entrenamiento

**Procesamiento lento:**
- **Nube:** La primera solicitud tarda 30-60s (despertar del servicio), luego es más rápida
- **Local sin GPU:** Se esperan 2-5s por video
- **Local con GPU:** Debería ser ~0,2s por video

**Error de conexión a la API:**
- **Nube:** Espera 60 segundos para que el servicio se active
- **Local:** Verifica que `python app.py` se esté ejecutando

**Error al subir video:**
- Verifica el formato del archivo (MP4, AVI, MOV compatibles)
- Asegúrate de que el tamaño del archivo sea < 100 MB
- Intenta convertir a MP4 con códec H.264

---

## ❓ Preguntas frecuentes

**P: ¿Necesito entrenar el modelo antes de usar el sistema?**  
**R:** ¡No! El sistema incluye un modelo preentrenado (`outputs/trained_model.pth`) listo para usar. Solo ejecuta `python app.py` y comienza a analizar videos.

**P: ¿Cuándo debería reentrenar el modelo?**  
**R:** Reentrena si:
- Tus cámaras muestran escenas muy diferentes (interior vs exterior, comercio vs estacionamiento)
- Estás obteniendo muchos falsos positivos o pasando por alto anomalías reales
- Necesitas adaptarlo a tu entorno específico

**P: ¿La exportación ONNX mejorará mis resultados?**  
**R:** No. La exportación ONNX NO cambia la precisión; es el mismo modelo en un formato diferente. Usa ONNX solo para:
- Implementación en dispositivos periféricos (Raspberry Pi, Jetson Nano)
- Plataformas que requieren formato ONNX (servicios cloud específicos)
- Optimizaciones específicas de hardware (TensorRT, OpenVINO)

Para hospedaje cloud normal o uso local, mantén el modelo de PyTorch.

**P: ¿Dónde obtengo el conjunto de datos UCSD Ped2?**  
**R:** Descárgalo desde la fuente oficial: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm  
El modelo actual ya está entrenado con este conjunto, por lo que solo lo necesitas si vas a reentrenar.

**P: ¿Cuántos datos necesito para reentrenar?**  
**R:** Mínimo 10-20 videos (30-60 segundos cada uno) de comportamiento normal. Cuanto más, mejor: apunta a 30+ minutos en total.

**P: ¿Puedo usar videos CON anomalías para el entrenamiento?**  
**R:** ¡No! Los datos de entrenamiento deben contener solo comportamiento normal. El modelo aprende cómo se ve lo "normal" y luego marca cualquier diferencia.

**P: ¿Cuánto tiempo tarda el entrenamiento?**  
**R:** 
- Con GPU (RTX 3050): 10-30 minutos
- Sin GPU (CPU): 2-4 horas
- Depende del tamaño del conjunto de datos y las épocas

**P: El sistema marca demasiados fotogramas normales como anomalías. ¿Qué hago?**  
**R:**
1. Aumenta el umbral usando el control deslizante del panel
2. Cambia al preajuste "Conservador"
3. Si persiste, reentrena con las grabaciones específicas de tu cámara

**P: El sistema pasa por alto anomalías evidentes. ¿Qué hago?**  
**R:**
1. Disminuye el umbral usando el control deslizante del panel
2. Cambia al preajuste "Sensible"
3. Verifica que tus anomalías coincidan con lo que el modelo fue entrenado para detectar (comportamiento peatonal)

---

## 🔗 Recursos adicionales

**Cómo funcionan los Autoencoders:**
- [Understanding Autoencoders](https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798)
- [Anomaly Detection with Autoencoders](https://arxiv.org/abs/1807.02108)

**Conjunto de datos UCSD Ped2:**
- [Dataset Information](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)
- Utilizado para entrenamiento y evaluación

**Tecnologías utilizadas:**
- **PyTorch** - Framework de aprendizaje profundo
- **FastAPI** - Framework de API REST
- **Streamlit** - Framework de paneles interactivos
- **OpenCV** - Procesamiento de video

---

## 📄 Licencia

[Apache License](LICENSE)

El conjunto de datos UCSD Ped2 se utiliza bajo licencia académica para el entrenamiento.

---

## 🤝 Contribuciones

¿Encontraste un error? ¿Tienes una sugerencia? Abre un issue en [GitHub](https://github.com/Aaryan2304/cctv-video-anomaly-detection/issues).

---

**Construido con ❤️ para aplicaciones de vigilancia y seguridad**
