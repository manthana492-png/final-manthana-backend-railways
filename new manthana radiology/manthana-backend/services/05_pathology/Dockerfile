# Minimal service-only image. For OpenRouter vision_pathology + full repo layout use Dockerfile.railway (build context=this_studio).
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
# OpenSlide: libopenslide for .svs, .ndpi, and other OpenSlide-backed formats.
# (.czi is not OpenSlide-native—use conversion or a CZI reader.) openslide-bin helps the Python binding locate libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libgl1 libglib2.0-0 \
    libopenslide-dev libopenslide0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8005
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
