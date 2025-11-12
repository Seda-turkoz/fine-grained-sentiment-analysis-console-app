# # import torch
# # import numpy as np
# # from sklearn.neural_network import MLPClassifier
# # from sklearn.metrics import classification_report, accuracy_score
# # from sklearn.model_selection import train_test_split

# # # ✅ Embedding ve label dosyalarını yükle
# # X = np.array(torch.load("trained_models/train_embeddings.pt"))  # Embedding'ler
# # y = np.array(torch.load("trained_models/train_labels.pt")
# #              )      # Etiketler (multi-label)

# # # ✅ Veriyi eğitim ve test olarak böl (örnek: %80 train, %20 test)
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X, y, test_size=0.2, random_state=42)

# # # ✅ MLPClassifier modelini tanımla (örnek yapı)
# # mlp = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu',
# #                     solver='adam', max_iter=20, random_state=42, verbose=True)

# # # 🔁 Modeli eğit
# # mlp.fit(X_train, y_train)

# # # 🔍 Tahmin yap
# # y_pred = mlp.predict(X_test)

# # # ✅ Değerlendirme
# # print("✅ Accuracy (Subset):", accuracy_score(y_test, y_pred))
# # print("\n📝 Classification Report:")
# # print(classification_report(y_test, y_pred, zero_division=0))


# import torch
# import numpy as np
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.model_selection import train_test_split
# import joblib
# import os

# # ✅ Yollar
# embedding_path = "trained_models/train_embeddings.pt"
# label_path = "trained_models/train_labels.pt"
# model_path = "trained_models/mlp_classifier_model.pkl"

# # ✅ Eğer model zaten varsa, yeniden eğitme
# if os.path.exists(model_path):
#     print("📦 MLP model already exists. Skipping training.")
# else:
#     print("🚀 No MLP model found. Training from scratch...")

#     # # 🔹 Verileri yükle
#     # X = np.array(torch.load(embedding_path))  # Embedding'ler
#     # y = np.array(torch.load(label_path))      # Etiketler

#     # # 🔹 Eğitim ve test bölme
#     # X_train, X_test, y_train, y_test = train_test_split(
#     #     X, y, test_size=0.2, random_state=42)


# # Embedding ve label dosyalarını yükle
# X_tensor_list = torch.load(embedding_path)
# X = np.stack([t.numpy() if isinstance(t, torch.Tensor)
#              else t for t in X_tensor_list])

# y_tensor = torch.load(label_path)
# y = y_tensor.numpy() if isinstance(y_tensor, torch.Tensor) else np.array(y_tensor)

# # 🔹 MLP modelini tanımla
# mlp = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu',
#                     solver='adam', max_iter=20, random_state=42, verbose=True)

# # 🔹 Eğit
# mlp.fit(X_train, y_train)

# # 🔹 Test ve sonuçlar
# y_pred = mlp.predict(X_test)
# print("✅ Accuracy (Subset):", accuracy_score(y_test, y_pred))
# print("\n📝 Classification Report:")
# print(classification_report(y_test, y_pred, zero_division=0))

# # 💾 Modeli kaydet
# joblib.dump(mlp, model_path)
# print(f"✅ MLP model saved to {model_path}")


def train_mlp_classifier_model():
    import torch
    import numpy as np
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.model_selection import train_test_split
    import joblib
    import os

    embedding_path = "trained_models/train_embeddings.pt"
    label_path = "trained_models/train_labels.pt"
    model_path = "trained_models/mlp_classifier_model.pkl"

    print("🚀 No MLP model found. Training from scratch...")

    # Embedding ve label dosyalarını yükle
    X_tensor_list = torch.load(embedding_path)
    X = np.stack([t.numpy() if isinstance(t, torch.Tensor)
                 else t for t in X_tensor_list])

    y_tensor = torch.load(label_path)
    y = y_tensor.numpy() if isinstance(y_tensor, torch.Tensor) else np.array(y_tensor)

    # Eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # MLP modelini tanımla ve eğit
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu',
                        solver='adam', max_iter=20, random_state=42, verbose=True)
    mlp.fit(X_train, y_train)

    # Performans çıktısı (opsiyonel ama faydalı)
    y_pred = mlp.predict(X_test)
    print("✅ Accuracy (Subset):", accuracy_score(y_test, y_pred))
    print("\n📝 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Modeli kaydet
    joblib.dump(mlp, model_path)
    print(f"✅ MLP model saved to {model_path}")
