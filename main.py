import os
import torch
import numpy as np
from rule_based.emotion_detector import detect_emotions_rule_based
from ai_based.logistic_regression.logistic_regression_predictor import predict_with_logistic
from ai_based.mlp.mlp_predictor import predict_with_mlp
from ai_based.logistic_regression_w_filter.logistic_attribution_predictor import predict_with_logistic_attribution


def print_menu():
    print("\n========= Emotion Detection Console App =========")
    print("1. Rule-Based Emotion Detection")
    print("2. Logistic Regression Classifier")
    print("3. MLP Classifier")
    print("4. Logistic Regression with Attribution Filter (coreferee-based) ")
    print("5. Exit")


def main():
    while True:
        print_menu()
        choice = input("\nSelect an option (1–5): ").strip()

        if choice == "1":
            text = input("\n📓 Enter your journal text:\n")
            result = detect_emotions_rule_based(text)
            print("\n🧠 Detected Emotions:", result)

        elif choice == "2":
            text = input("\n📓 Enter your journal text:\n")
            emotions = predict_with_logistic(text)
            print(f"\n🧠 Predicted Emotions (Logistic Regression): {emotions}")
            print("📊 Pie chart saved and displayed.")
            input("🔁 Press ENTER to return to the main menu...")

        elif choice == "3":
            text = input("\n📓 Enter your journal text:\n")
            result = predict_with_mlp(text)
            print("\n🧠 Predicted Emotions (MLP):", result)
            print("📊 Pie chart saved and displayed.")
            input("🔁 Press ENTER to return to the main menu...")

        elif choice == "4":
            text = input("\n📓 Enter your journal text:\n")
            result = predict_with_logistic_attribution(text)
            print("\n🧠 Predicted Emotions (Logistic Regression w Filtered):", result)
            print("📊 Pie chart saved and displayed.")
            input("🔁 Press ENTER to return to the main menu...")

        elif choice == "5":
            print("\n👋 Exiting. Goodbye.")
            break

        else:
            print("\n❌ Invalid selection. Choose between 1 and 5.")


if __name__ == "__main__":
    main()
