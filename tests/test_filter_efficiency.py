# import os
# import pandas as pd
# import re
# import random
# import torch
# import joblib
# import spacy
# from ai_based.embedding_pipeline import get_embedding_from_text
# from utils.label_map import id2emotion
# import coreferee  # Bunu da unutma!

# # 🔹 1. CSV dosyalarını oku
# df1 = pd.read_csv("archive/data/full_dataset/goemotions_1.csv")
# df2 = pd.read_csv("archive/data/full_dataset/goemotions_2.csv")
# df3 = pd.read_csv("archive/data/full_dataset/goemotions_3.csv")

# full_df = pd.concat([df1, df2, df3], ignore_index=True)
# texts = full_df["text"].dropna().tolist()

# # 🔹 2. Regex tanımları
# first_person = re.compile(r"\b(I|me|my|mine|myself)\b", re.IGNORECASE)
# third_person = re.compile(
#     r"\b(he|she|his|her|they|them|their|friend|mom|dad|teacher|sister|brother|boss|coworker|my\s\w+)\b", re.IGNORECASE)

# # 🔹 3. Setleri oluştur
# set_a, set_b, set_c = [], [], []

# for text in texts:
#     has_first = bool(first_person.search(text))
#     has_third = bool(third_person.search(text))

#     if has_first and not has_third:
#         set_a.append(text)
#     elif has_first and has_third:
#         set_b.append(text)
#     elif has_third and not has_first:
#         set_c.append(text)

# # 🔹 4. Örnekle 300 cümle al (yoksa tamamını al)
# sample_a = random.sample(set_a, min(300, len(set_a)))
# sample_b = random.sample(set_b, min(300, len(set_b)))
# sample_c = random.sample(set_c, min(300, len(set_c)))

# # Dataset sözlüğünü tanımla
# datasets = {
#     "Set A (1st person only)": sample_a,
#     "Set B (Mixed)": sample_b,
#     "Set C (3rd person only)": sample_c
# }

# # 🔹 5. Örnek çıktı
# print(f"Set A (1st person only): {len(sample_a)} samples")
# print(f"Set B (Mixed): {len(sample_b)} samples")
# print(f"Set C (3rd person only): {len(sample_c)} samples")


# # ------------------------------------------------------------------------------------------

# # Kaydetmek için klasör oluştur
# os.makedirs("tests/test_sets", exist_ok=True)

# # Setleri dışa aktar
# file_map = {
#     "Set A (1st person only)": "set_a_1st_person.txt",
#     "Set B (Mixed)": "set_b_mixed.txt",
#     "Set C (3rd person only)": "set_c_3rd_person.txt"
# }

# for name, samples in datasets.items():
#     path = os.path.join("tests", "test_sets", file_map[name])
#     with open(path, "w", encoding="utf-8") as f:
#         for line in samples:
#             f.write(line.strip() + "\n")

# print("\n✅ Test set dosyaları 'tests/test_sets/' klasörüne kaydedildi.")


# # -----------------------------------------------------------------------------------------


# # 🔹 Modeli yükle
# model = joblib.load("trained_models/logisticregression_model.pkl")

# # 🔹 spaCy + coreferee başlat
# nlp = spacy.load("en_core_web_sm")
# nlp.add_pipe("coreferee")


# # 🔹 Attribution filter fonksiyonu


# def is_emotion_from_author(text):
#     doc = nlp(text)
#     if doc._.has_coref:
#         for chain in doc._.coref_chains:
#             for mention in chain:
#                 if mention.text.lower() in ["i", "me", "my", "myself"]:
#                     return True
#         return False
#     else:
#         return any(tok.lower_ in ["i", "me", "my", "myself"] for tok in doc)


# # 🔹 Setleri sözlük haline getir
# datasets = {
#     "Set A (1st person only)": sample_a,
#     "Set B (Mixed)": sample_b,
#     "Set C (3rd person only)": sample_c
# }

# print("\n📊 Evaluation Results:")
# print("| Set Name             | Total | Predicted | Passed Filter | Filtered Out | Retention % |")
# print("|----------------------|-------|-----------|----------------|---------------|--------------|")


# for name, samples in datasets.items():
#     predicted = 0
#     passed_filter = 0

#     for text in samples:
#         embedding = get_embedding_from_text(text)
#         if embedding is None:
#             continue

#         prediction = model.predict(torch.tensor(embedding).reshape(1, -1))
#         # emotion_id = int(prediction[0])
#         # emotion_id = prediction[0].item()
#         emotion_id = int(prediction[0][0])
#         emotion = id2emotion.get(emotion_id, None)

#         if emotion:
#             predicted += 1
#             if is_emotion_from_author(text):
#                 passed_filter += 1

#     filtered_out = predicted - passed_filter
#     retention_rate = (passed_filter / predicted * 100) if predicted > 0 else 0
#     print(f"| {name:<21} | {len(samples):<5} | {predicted:<9} | {passed_filter:<14} | {filtered_out:<13} | {retention_rate:>10.1f}% |")


# import os
# import pandas as pd
# import re
# import random
# import torch
# import joblib
# import spacy
# import coreferee  # Bu çok önemli, başta import edilmeli
# from ai_based.embedding_pipeline import get_embedding_from_text
# from utils.label_map import id2emotion

# # 🔹 1. CSV dosyalarını oku
# df1 = pd.read_csv("archive/data/full_dataset/goemotions_1.csv")
# df2 = pd.read_csv("archive/data/full_dataset/goemotions_2.csv")
# df3 = pd.read_csv("archive/data/full_dataset/goemotions_3.csv")

# full_df = pd.concat([df1, df2, df3], ignore_index=True)
# texts = full_df["text"].dropna().tolist()

# # 🔹 2. Regex tanımları
# first_person = re.compile(r"\b(I|me|my|mine|myself)\b", re.IGNORECASE)
# third_person = re.compile(
#     r"\b(he|she|his|her|they|them|their|friend|mom|dad|teacher|sister|brother|boss|coworker|my\s\w+)\b", re.IGNORECASE)

# # 🔹 3. Setleri oluştur
# set_a, set_b, set_c = [], [], []

# for text in texts:
#     has_first = bool(first_person.search(text))
#     has_third = bool(third_person.search(text))

#     if has_first and not has_third:
#         set_a.append(text)
#     elif has_first and has_third:
#         set_b.append(text)
#     elif has_third and not has_first:
#         set_c.append(text)

# # 🔹 4. Örnekle 300 cümle al (yoksa tamamını al)
# sample_a = random.sample(set_a, min(300, len(set_a)))
# sample_b = random.sample(set_b, min(300, len(set_b)))
# sample_c = random.sample(set_c, min(300, len(set_c)))

# datasets = {
#     "Set A (1st person only)": sample_a,
#     "Set B (Mixed)": sample_b,
#     "Set C (3rd person only)": sample_c
# }

# print(f"Set A (1st person only): {len(sample_a)} samples")
# print(f"Set B (Mixed): {len(sample_b)} samples")
# print(f"Set C (3rd person only): {len(sample_c)} samples")

# # 🔹 5. Kaydet
# os.makedirs("tests/test_sets", exist_ok=True)
# file_map = {
#     "Set A (1st person only)": "set_a_1st_person.txt",
#     "Set B (Mixed)": "set_b_mixed.txt",
#     "Set C (3rd person only)": "set_c_3rd_person.txt"
# }

# for name, samples in datasets.items():
#     path = os.path.join("tests", "test_sets", file_map[name])
#     with open(path, "w", encoding="utf-8") as f:
#         for line in samples:
#             f.write(line.strip() + "\n")

# print("\n✅ Test set dosyaları 'tests/test_sets/' klasörüne kaydedildi.")

# # 🔹 Modeli yükle
# model = joblib.load("trained_models/logisticregression_model.pkl")

# # 🔹 spaCy + coreferee başlat
# nlp = spacy.load("en_core_web_sm")
# if not nlp.has_pipe("coreferee"):
#     nlp.add_pipe("coreferee")  # Bu kısım çok önemli ❗

# # 🔹 Attribution filter


# def is_emotion_from_author(text):
#     doc = nlp(text)
#     if doc._.has_coref:
#         for chain in doc._.coref_chains:
#             for mention in chain:
#                 if mention.text.lower() in ["i", "me", "my", "myself"]:
#                     return True
#         return False
#     else:
#         return any(tok.lower_ in ["i", "me", "my", "myself"] for tok in doc)


# # 🔹 Değerlendirme
# print("\n📊 Evaluation Results:")
# print("| Set Name             | Total | Predicted | Passed Filter | Filtered Out | Retention % |")
# print("|----------------------|-------|-----------|----------------|---------------|--------------|")

# for name, samples in datasets.items():
#     predicted = 0
#     passed_filter = 0

#     for text in samples:
#         embedding = get_embedding_from_text(text)
#         if embedding is None:
#             continue

#         # prediction = model.predict(torch.tensor(embedding).reshape(1, -1))
#         prediction = model.predict(embedding.reshape(1, -1))
#         # emotion_id = int(prediction[0])  # Veya prediction[0][0] gibi değil!
#         emotion_id = int(prediction[0]) if prediction.ndim == 1 else int(
#             prediction[0][0])
#         emotion = id2emotion.get(emotion_id, None)

#         if emotion:
#             predicted += 1
#             if is_emotion_from_author(text):
#                 passed_filter += 1

#     filtered_out = predicted - passed_filter
#     retention_rate = (passed_filter / predicted * 100) if predicted > 0 else 0
#     print(f"| {name:<21} | {len(samples):<5} | {predicted:<9} | {passed_filter:<14} | {filtered_out:<13} | {retention_rate:>10.1f}% |")
# ----------------------------------------------------------------------------------------------------------------------------------------------------seda

# import json
# from pathlib import Path
# from ai_based.logistic_regression_w_filter.logistic_attribution_predictor import predict_with_logistic_attribution, is_emotion_from_author_v2


# # Test dosyalarının yolu
# test_dir = Path("tests/test_sets")

# set_files = {
#     "Set A (1st person only)": test_dir / "set_a_1st_person.txt",
#     "Set B (Mixed)": test_dir / "set_b_mixed.txt",
#     "Set C (3rd person only)": test_dir / "set_c_3rd_person.txt"
# }

# print("\n📊 Evaluation Results:")
# print("| Set Name             | Total | Predicted | Passed Filter | Filtered Out | Retention % |")
# print("|----------------------|-------|-----------|----------------|---------------|--------------|")

# for set_name, file_path in set_files.items():
#     with open(file_path, "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     total = len(lines)
#     predicted = 0
#     passed_filter = 0


# for line in lines:
#     text = line.strip()  # artık sadece düz metin

#     if not text:
#         continue  # boş satır varsa geç

#     if is_emotion_from_author_v2(text):
#         passed_filter += 1
#         emotions = predict_with_logistic_attribution(text)
#         if emotions and "[No Emotion Detected]" not in emotions and "[Filtered: Not Author's Emotion]" not in emotions:
#             predicted += 1

#     # for line in lines:
#     #     data = json.loads(line)
#     #     text = data["text"]

#     #     # Attribution filtresi geçerse prediction yapılır
#     #     if is_emotion_from_author_v2(text):
#     #         passed_filter += 1
#     #         emotions = predict_with_logistic_attribution(text)
#     #         if emotions and "[No Emotion Detected]" not in emotions and "[Filtered: Not Author's Emotion]" not in emotions:
#     #             predicted += 1

#     filtered_out = total - passed_filter
#     retention_pct = (passed_filter / total) * 100

#     print(f"| {set_name:<20} | {total:5} | {predicted:9} | {passed_filter:14} | {filtered_out:13} | {retention_pct:10.2f}% |")

# çok ağır geldi çalışmadı
# import os
# import pandas as pd
# import random
# import re
# from ai_based.logistic_regression_w_filter.logistic_attribution_predictor import is_emotion_from_author_v2

# # 🔹 1. Büyük dataset (GoEmotions CSV'leri)
# df1 = pd.read_csv("archive/data/full_dataset/goemotions_1.csv")
# df2 = pd.read_csv("archive/data/full_dataset/goemotions_2.csv")
# df3 = pd.read_csv("archive/data/full_dataset/goemotions_3.csv")

# full_df = pd.concat([df1, df2, df3], ignore_index=True)
# texts = full_df["text"].dropna().tolist()

# # 🔹 2. Attribution Layer ile Author-Related ve Not Author-Related ayır
# author_related_list = [t for t in texts if is_emotion_from_author_v2(t)]
# not_author_related_list = [
#     t for t in texts if not is_emotion_from_author_v2(t)]

# print(f"✅ Found {len(author_related_list)} Author-Related and {len(not_author_related_list)} Not Author-Related samples.")

# # 🔹 3. Dengeli 150 + 150 örnek seç
# balanced_set_b = random.sample(author_related_list, min(150, len(author_related_list))) + \
#     random.sample(not_author_related_list, min(
#         150, len(not_author_related_list)))

# random.shuffle(balanced_set_b)

# # 🔹 4. Yeni dosyayı kaydet
# os.makedirs("tests/test_sets", exist_ok=True)
# output_path = "tests/test_sets/set_b_mixed_balanced.txt"

# with open(output_path, "w", encoding="utf-8") as f:
#     for sentence in balanced_set_b:
#         f.write(sentence.strip() + "\n")

# print(f"✅ Balanced Set B created and saved to: {output_path}")


# seda

import json
from pathlib import Path
from ai_based.logistic_regression_w_filter.logistic_attribution_predictor import predict_with_logistic_attribution, is_emotion_from_author_v2

test_dir = Path("tests/test_sets")
set_files = {
    "Set A (1st person only)": test_dir / "set_a_1st_person.txt",
    # "Set B (Mixed)": test_dir / "set_b_mixed.txt",
    "Set B (Mixed)": test_dir / "set_b_mixed_balanced.txt",
    "Set C (3rd person only)": test_dir / "set_c_3rd_person.txt",
}

print("\n📊 Evaluation Results:")
print("| Set Name             | Total | Author-Related  | Not Author-Related| Predicted  | Retention % |")
print("|----------------------|-------|-----------------|-------------------|------------|-------------|")

for set_name, file_path in set_files.items():
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    author_related = 0
    predicted = 0

    for line in lines:
        text = line.strip()
        if not text:
            continue

        # ✅ Önce Attribution Filter uygula
        if is_emotion_from_author_v2(text):
            author_related += 1
            # ✅ Sadece filtreyi geçenleri sınıflandır
            prediction = predict_with_logistic_attribution(text)
            if prediction:
                predicted += 1

    # ✅ Doğru metrik hesaplamaları
    not_author_related = total - author_related
    retention_rate = (author_related / total * 100) if total else 0

    print(f"| {set_name:<21} | {total:5} | {author_related:9} | {not_author_related :14} | {predicted:13} | {retention_rate:10.2f}% |")
# seda
