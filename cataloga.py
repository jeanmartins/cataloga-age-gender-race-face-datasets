import os
import cv2
import pandas as pd
from deepface import DeepFace

PASTA_TESTES = os.environ.get("DATASET_PATH")

stats = {
    "imagens_invalidas": 0,
    "sem_face": 0,
    "homens": 0,
    "mulheres": 0,
    "idade_0_6": 0,
    "idade_7_12": 0,
    "idade_13_20": 0,
    "idade_21_30": 0,
    "idade_31_45": 0,
    "idade_46_60": 0,
    "idade_61_100": 0,
}

for root, dirs, files in os.walk(PASTA_TESTES):
    for arquivo in files:
        if not arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        caminho = os.path.join(root, arquivo)
        img = cv2.imread(caminho)
        if img is None:
            print(f"[WARN] Falha ao ler imagem: {caminho}")
            stats["imagens_invalidas"] += 1
            stats["sem_face"] += 1
            continue
        analysis = DeepFace.analyze(img, actions=["age", "gender"], enforce_detection=False)
        if isinstance(analysis, dict):
            analysis = [analysis]

        if len(analysis) == 0:
            stats["sem_face"] += 1
            continue
        
        for face in analysis:
            g = face["gender"]
            idade = face["age"]

            if g["Man"] > g["Woman"]:
                stats["homens"] += 1
            else:
                stats["mulheres"] += 1

            if idade < 7:
                stats["idade_0_6"] += 1
            elif idade < 13:
                stats["idade_7_12"] += 1
            elif idade < 21:
                stats["idade_13_20"] += 1
            elif idade < 31:
                stats["idade_21_30"] += 1
            elif idade < 46:
                stats["idade_31_45"] += 1
            elif idade < 61:
                stats["idade_46_60"] += 1
            elif idade < 101:
                stats["idade_61_100"] += 1
            

print(stats)
df = pd.DataFrame([stats])
df.to_csv('stats.csv', index=False)