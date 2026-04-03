FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/* && ln -s /usr/bin/python3 /usr/bin/python
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    "git+https://github.com/MIC-DKFZ/nnUNet.git@v2.4.2#egg=nnunetv2"
RUN pip install --no-cache-dir -r requirements.txt
RUN git clone --depth 1 https://github.com/StanfordMIMI/Comp2Comp /opt/Comp2Comp \
    && cd /opt/Comp2Comp \
    && pip install --no-cache-dir -e . \
    && echo "Comp2Comp installed: $(./bin/C2C --version 2>&1 || echo installed)"
ENV COMP2COMP_DIR=/opt/Comp2Comp
COPY . .
EXPOSE 8008
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8008"]
