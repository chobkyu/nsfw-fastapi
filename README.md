git lfs install

git clone https://huggingface.co/Falconsai/nsfw_image_detection

python -m venv venv

venv\Scripts\activate  # 가상환경 진입

pip install -r requirements.txt

uvicorn app:app --reload
