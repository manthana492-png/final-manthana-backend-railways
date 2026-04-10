FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "\
import torchxrayvision as xrv; \
xrv.models.DenseNet(weights='densenet121-res224-all'); \
xrv.models.DenseNet(weights='densenet121-res224-chex'); \
xrv.models.DenseNet(weights='densenet121-res224-mimic_nb'); \
print('TXRV weights pre-downloaded OK')"

COPY . .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
