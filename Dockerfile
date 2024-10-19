# Menggunakan image dasar Python 3.10
FROM python:3.10

# Mengatur direktori kerja di dalam container
WORKDIR /app

# Menyalin file requirements.txt ke dalam container
COPY requirements.txt .

# Menginstal dependensi yang diperlukan
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh kode sumber ke dalam container
COPY . .

# Menjalankan aplikasi
CMD ["python", "app.py"]
