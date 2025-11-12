import matplotlib.pyplot as plt
import os


def plot_emotion_pie(emotion_labels, emotion_scores, model_name="model"):
    plt.figure(figsize=(6, 6))
    plt.pie(emotion_scores, labels=emotion_labels,
            autopct='%1.1f%%', startangle=140)
    plt.title(f'Predicted Emotion Distribution ({model_name})')
    plt.axis('equal')
    plt.tight_layout()

    # Save with dynamic name
    save_dir = "visuals"
    os.makedirs(save_dir, exist_ok=True)  # klasör yoksa oluştur
    save_path = os.path.join(
        save_dir, f"{model_name.lower().replace(' ', '_')}_pie_chart.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Pie chart saved as: {save_path}")

    plt.show()
