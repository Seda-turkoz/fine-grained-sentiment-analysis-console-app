
import re
from collections import defaultdict
import matplotlib.pyplot as plt

# Genişletilmiş emotion keyword dictionary
emotion_keywords = {



    "sadness": [
        "sad", "unhappy", "mournful", "teary", "lonely", "crying", "hopeless", "heartbroken", "lost",
        "broken", "grief", "melancholy", "low", "down", "blue", "depressed", "painful", "dismal", "defeated",
        "empty", "gloomy", "shattered", "isolated", "hurt", "dark", "downhearted", "distressed", "hopeless",
        "mourn", "sorrow", "empty inside", "feeling lost", "tired of life", "numb", "feeling abandoned",
        "dark day", "cold night", "silent room", "alone in bed", "can't stop crying",
        "felt abandoned", "missed them so much", "nothing feels right", "rainy weather",
        "silent tears", "overwhelming sadness", "empty inside", "life feels heavy"
    ],


    "happiness": [
        "happy", "joy", "joyful", "cheerful", "smile", "laugh", "grateful", "thankful", "pleased", "satisfied",
        "content", "excited", "bubbly", "bright", "sunny", "radiant", "fun", "lighthearted", "elated",
        "uplifted", "thrilled", "delighted", "positive", "high spirits", "energetic", "overjoyed",
        "feel great", "so good", "bursting with joy", "full of energy", "loving life", "on cloud nine", "pure bliss", "smiling", "laughter", "peace", "relaxed", "calm", "sunlight", "walk in the park",
        "children playing", "kids playing", "fresh air", "beautiful day", "nature", "flowers",
        "smile on my face", "seeing people smile", "felt peaceful", "felt relaxed", "positive vibes",
        "good weather", "morning coffee", "nice atmosphere", "birds singing"
    ],


    "anxiety": [
        "anxious", "worried", "nervous", "panicked", "restless", "jittery", "uneasy", "shaky", "tense",
        "uncertain", "unsettled", "overwhelmed", "fearful", "dizzy", "clammy", "racing heart", "on edge",
        "can't focus", "expecting the worst", "uneasy feeling", "trouble breathing", "tight chest", "can't relax",
        "afraid of future", "irrational fear", "can't sleep", "sweating", "feeling trapped", "mind racing",
        "can't sleep", "heart was racing", "felt trapped", "mind keeps spinning",
        "trouble breathing", "can't relax", "expecting the worst", "panic attack",
        "fear of future", "shaky hands", "dizzy feeling", "pressure in chest",
        "worried all night", "overthinking everything"
    ],


    "hope": [
        "hope", "believe", "faith", "optimistic", "possibility", "future", "light", "chance", "dream",
        "aspiration", "wishful", "anticipation", "recovery", "rebuilding", "positive outcome", "improvement",
        "progress", "perseverance", "determination", "goal", "persistence", "vision", "looking ahead", "light at the end",
        "healing", "stay strong", "bright future", "stronger tomorrow", "goal-oriented", "expectation", "keep fighting",
        "enduring", "keep moving forward", "never give up", "facing tomorrow", "better days ahead", "light at the end", "things will get better",
        "new beginning", "fresh start", "not giving up", "found my way",
        "holding on", "keep going", "stay hopeful", "I still believe",
        "there is hope", "hopeful thoughts", "never lose faith",
        "better days ahead", "light at the end", "things will get better",
        "new beginning", "fresh start", "still believe", "not giving up",
        "holding on", "positive future", "stay hopeful", "looking forward",
        "hopeful thoughts", "brighter days coming"
    ],




    "anger": [
        "angry", "furious", "irritated", "annoyed", "resentful", "frustrated", "mad", "infuriated", "rage",
        "boiling", "fuming", "enraged", "pissed off", "exploded", "lost control", "can't stand it", "snapped",
        "burning inside", "gritted teeth", "stormed out", "feeling rage", "about to explode", "burst into anger",
        "lost temper", "lashed out", "overreacted", "hurtful words", "aggressive", "felt like yelling", "lost control", "raised voice", "shouting loudly", "slammed the door", "felt like punching",
        "burning inside", "can't stand it", "exploded in anger", "lashed out", "felt insulted",
        "argument broke out", "stormed out", "threw something", "couldn't calm down"
    ],


    "numbness": [
        "numb", "empty", "disconnected", "emotionless", "blank", "robotic", "shut down", "frozen", "void",
        "detached", "apathetic", "unmoved", "silent", "cold", "flat", "dull", "without feeling",
        "empty inside", "feeling dead", "feel nothing", "can't feel", "without emotions", "shut myself down",
        "lost all feeling", "detached from life", "silent pain", "just existing", "hollow", "emotionally dead",
        "felt nothing", "just existing", "emotionally dead", "can't feel anything",
        "empty inside", "lost all feeling", "shut down", "disconnected from life",
        "without emotions", "hollow feeling", "no reaction", "moving like a robot",
        "silent pain"
    ],





    "guilt": [
        "guilty", "regret", "ashamed", "sorry", "remorse", "blame", "fault", "apologize", "confess",
        "wrong", "mistake", "burdened", "haunted", "sin", "forgive me", "wish I hadn't", "can't forgive myself",
        "feel responsible", "my fault", "feel bad", "hard to forgive", "troubled by", "regretting", "self-blame",
        "feeling bad", "deep regret", "haunted by past", "bad decision", "own mistake", "internal blame",
        "can't forgive myself", "wish I hadn't done that", "it was my fault",
        "made a mistake", "feel responsible", "hard to forgive", "regretting deeply",
        "troubled by past", "feeling bad about it", "can't undo it", "haunted by mistake",
        "carrying guilt", "weighed down by guilt"
    ],


    "shame": [
        "shame", "ashamed", "humiliated", "embarrassed", "worthless", "inferior", "awkward", "disgraced",
        "unworthy", "self-conscious", "cringe", "mortified", "exposed", "feeling small", "publicly embarrassed",
        "hiding face", "can’t look", "judged", "bad reputation", "unacceptable", "felt stupid", "loss of respect",
        "disapproval", "blushed", "looked down", "hiding", "felt naked", "too embarrassed", "can't face people",
        "felt so stupid", "wanted to hide", "hiding face", "can't look in the eyes",
        "publicly embarrassed", "loss of respect", "felt judged", "felt small",
        "looked down on", "people staring", "can't face them", "feeling exposed"
    ],


    "frustration": [
        "frustrated", "stuck", "annoyed", "blocked", "helpless", "tired", "fed up", "angry", "irritated",
        "overwhelmed", "burned out", "no progress", "slow progress", "exhausted", "pressure", "can't move on",
        "losing patience", "worn out", "mentally tired", "repeating mistakes", "going nowhere", "hitting a wall",
        "losing control", "can't handle", "drained", "emotionally exhausted", "sick of this", "done with this", "can't stand it",
        "nothing works", "can't fix this", "can't move on", "hitting a wall", "tried so hard",
        "losing patience", "mentally tired", "drained energy", "stuck in a loop", "wasting time",
        "can't handle anymore", "keep failing", "overwhelming feeling", "burned out"
    ],



    "gratitude": [
        "grateful", "thankful", "appreciate", "blessed", "content", "glad", "happy for", "recognized", "valued",
        "honored", "fortunate", "deep thanks", "so thankful", "full of gratitude", "gifted", "cherish", "meaningful",
        "feel lucky", "life is good", "thank you", "grateful heart", "appreciation", "warm feeling", "thankful for them",
        "so lucky to have", "couldn’t have done it without", "thanks a lot",
        "feeling thankful", "full of gratitude", "appreciate their help",
        "feeling blessed to know", "beyond grateful", "feeling supported",
        "warm heart", "lucky to be here", "thankful for this moment"
    ],



    "fear": [
        "fear", "afraid", "scared", "terrified", "worried", "shaky", "anxious", "dread", "paranoid",
        "frightened", "startled", "nervous", "panicked", "frozen", "horrified", "uneasy", "spooked",
        "fearful", "insecure", "unsafe", "alert", "vulnerable", "trembling", "nightmare",
        "fear of failure", "fear of rejection", "fear of future", "panic attack", "can't breathe", "cold sweat",
        "fear of losing", "fear of future", "can't breathe", "frozen with fear", "nightmare",
        "dark place", "strange noise", "felt threatened", "felt unsafe", "insecure feeling",
        "on edge", "heart pounding", "hiding from fear", "afraid to move", "couldn't speak"
    ],




    "joy": [
        "joy", "delight", "bliss", "happiness", "euphoric", "thrilled", "cheerful", "excited", "laughter",
        "overjoyed", "festive", "glowing", "vibrant", "ecstatic", "sunshine", "bubbling with joy", "jumping with joy",
        "pure happiness", "can't stop smiling", "feeling alive", "energetic", "celebrating", "over the moon",
        "sparkling", "happy tears", "radiating happiness", "greatest day", "high spirits", "best feeling",
        "jumping with joy", "bursting with energy", "can't stop smiling",
        "best day ever", "feeling alive", "over the moon", "so much fun",
        "high spirits", "laughing out loud", "sunny day", "felt amazing",
        "full of joy", "hearts filled with joy"
    ]

}


# Pie Chart (Yüzdelik Duygu Dağılımı) Oluştur ve Kaydet
def generate_pie_chart(emotion_counts):
    if not emotion_counts:
        print("No emotions detected to visualize.")
        return

    labels = emotion_counts.keys()
    sizes = emotion_counts.values()

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Detected Emotions Distribution")
    plt.axis('equal')  # Daire şeklinde olsun

    # PNG olarak kaydet
    plt.savefig("emotion_output.png")
    print("Pie Chart saved as 'emotion_output.png'.")

    # İstersen grafik ekranda göster
    plt.show()


def detect_emotions_rule_based(text_input):
    words = re.findall(r'\b\w+\b', text_input.lower())

    emotion_counts = defaultdict(int)
    for emotion, keywords in emotion_keywords.items():
        for word in words:
            if word in keywords:
                emotion_counts[emotion] += 1

    sorted_emotions = sorted(emotion_counts.items(),
                             key=lambda x: x[1], reverse=True)

    print("\n🧠 Detected Emotions:")
    if sorted_emotions and sorted_emotions[0][1] > 0:
        for emotion, count in sorted_emotions:
            if count > 0:
                print(f"- {emotion.capitalize()}: {count}")
        print(f"\n🏆 Most dominant emotion: {sorted_emotions[0][0].upper()}")
        generate_pie_chart(emotion_counts)
    else:
        print("No emotions detected.")

    return dict(emotion_counts)


if __name__ == "__main__":
    print("📓 Enter your journal text (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    text_input = " ".join(lines)
    detect_emotions_rule_based(text_input)
