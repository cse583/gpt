import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

load_dotenv()

client = OpenAI()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY NOT FOUND")

client.api_key = OPENAI_API_KEY

def chat_with_gpt(prompt):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a master of hotpath detection. You will get a series of IR representing a whole path, "
                        "and you will ONLY provide a confidence score between 0 and 1 for your decision in the format: "
                        "'this is a hot path, confidence: X' or 'this is not a hot path, confidence: X'. replace X with the confidence value. warning: don't output any extra things other than what I told you to output"
                    )
                },
                {"role": "user", "content": f"{prompt}"}
            ]
        )
        response = completion.choices[0].message.content.strip().lower()
        
        if "confidence:" in response:
            parts = response.split("confidence:")
            confidence = float(parts[1].strip())
            
            if "this is a hot path" in response:
                return confidence 
            elif "this is not a hot path" in response:
                return 1 - confidence 
        else:
            return None 
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    file_path = "test-00000-of-00001.csv"
    df = pd.read_csv(file_path)

    true_labels = []
    predicted_scores = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        path = row["path"]
        label = row["label"]
        gpt_response = chat_with_gpt(path)

        if gpt_response is not None:
            true_labels.append(label)
            predicted_scores.append(gpt_response)
        else:
            print(f"Warning: No valid response for row {index}")
            true_labels.append(label)
            predicted_scores.append(0.5)

    predicted_classes = [1 if score >= 0.5 else 0 for score in predicted_scores]

    accuracy = accuracy_score(true_labels, predicted_classes)
    auroc = roc_auc_score(true_labels, predicted_scores)
    f1 = f1_score(true_labels, predicted_classes)

    print(f"Accuracy: {accuracy:.2%}")
    print(f"AUROC: {auroc:.2f}")
    print(f"F1 Score: {f1:.2f}")

    result_df = pd.DataFrame({
        "path": df["path"],
        "label": true_labels,
        "predicted_score": predicted_scores,
        "predicted_class": predicted_classes
    })
    result_df.to_csv("gpt_path_results_with_confidence.csv", index=False)
    print("Results saved to 'gpt_path_results_with_confidence.csv'")
