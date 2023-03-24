# Uncomment lines below to download VADER dictionary needed for analysis. Downlaod is only needed once.
# import nltk
# nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv

RESULTS_DIR_PATH = "results/"
results_count = {"positive": 0, "neutral": 0, "negative": 0, "total_prompts": 0}

# Analyze each prompt from the image results csv file
with open(
    file=RESULTS_DIR_PATH + "image_results.csv", mode="r", encoding="utf-8"
) as img_csv_file:
    with open(
        file="prompt_results.csv", mode="w", encoding="utf-8", newline=""
    ) as prompt_csv_file:
        reader = csv.reader(img_csv_file)
        writer = csv.writer(prompt_csv_file)
        analyzer = SentimentIntensityAnalyzer()

        for row in reader:
            file_id = row[0]
            prompt = row[1]
            # Remove hashtags since analyzer treats words preceded by hashtags as neutral
            scores = analyzer.polarity_scores(prompt.replace("#", ""))
            compound = scores["compound"]

            sentiment = ""
            if compound >= 0.05:
                sentiment = "positive"
            elif compound <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            # Write sentiment and scores for each prompt to CSV
            results_count[sentiment] += 1
            results_count["total_prompts"] += 1
            writer.writerow([file_id, prompt, sentiment, compound, scores])

# Write tally of prompt sentiments detected to CSV
print("Writing summary of results...")
with open(
    file=RESULTS_DIR_PATH + "prompt_results_summary.csv", mode="w", newline=""
) as csv_file:
    writer = csv.writer(csv_file)
    for key in results_count.keys():
        writer.writerow([key, results_count[key]])

print("Done! " + str(results_count["total_prompts"]) + " prompts analyzed.")
