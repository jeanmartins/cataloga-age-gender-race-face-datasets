import os
import cv2
from insightface.app import FaceAnalysis
import pandas as pd
from deepface import DeepFace

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0,det_size=(320,320))

PASTA_TESTES = '/mnt/hdexterno/VGGFACE2/'

stats = {
    "homens": 0,
    "mulheres": 0,
    "idade_0_6": 0,
    "idade_7_12": 0,
    "idade_13_20": 0,
    "idade_21_30": 0,
    "idade_31_45": 0,
    "idade_46_60": 0,
    "idade_61_100": 0,
    "emocao": {},
    "racas": {}
}

for root, dirs, files in os.walk(PASTA_TESTES):
    for arquivo in files:
        if not arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        caminho = os.path.join(root, arquivo)
        img = cv2.imread(caminho)
        faces = app.get(img)
        analysis = DeepFace.analyze(img, actions=["age", "gender", "race", "emotion"], enforce_detection=False)

        for face in faces:
            g = face["gender"]
            idade = face["age"]

            if g == 1:
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


        for face in analysis:
            emotion = face["dominant_emotion"]
            raca = face["dominant_race"]

            stats["emocao"][emotion] = stats["emocao"].get(emotion, 0) + 1
            stats["racas"][raca] = stats["racas"].get(raca, 0) + 1

print(stats)
df = pd.DataFrame([stats])
df.to_csv('stats.csv', index=False)