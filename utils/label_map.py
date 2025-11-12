import csv
import json

# Label mapping from GoEmotions
label_map = {
    "0": "admiration", "1": "amusement", "2": "anger", "3": "annoyance", "4": "approval",
    "5": "caring", "6": "confusion", "7": "curiosity", "8": "desire", "9": "disappointment",
    "10": "disapproval", "11": "disgust", "12": "embarrassment", "13": "excitement", "14": "fear",
    "15": "gratitude", "16": "grief", "17": "joy", "18": "love", "19": "nervousness",
    "20": "optimism", "21": "pride", "22": "realization", "23": "relief", "24": "remorse",
    "25": "sadness", "26": "surprise", "27": "neutral"
}

# ID to emotion string mapping (used in prediction)
id2emotion = {int(k): v for k, v in label_map.items()}

input_file = 'archive/data/test.tsv'
output_file = 'archive/data/test_preprocessed.jsonl'


# Her satırı işle
with open(input_file, 'r', encoding='utf-8') as tsv_in, open(output_file, 'w', encoding='utf-8') as json_out:
    reader = csv.reader(tsv_in, delimiter='\t')
    for row in reader:
        if len(row) != 3:
            continue  # hatalı satır varsa atla
        text, label_ids, _ = row
        label_names = [label_map[label_id]
                       for label_id in label_ids.split(',')]
        json.dump({"text": text, "labels": label_names}, json_out)
        json_out.write('\n')

print("✅ Conversion complete! File saved as:", output_file)
