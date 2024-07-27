from difflib import SequenceMatcher

# List of anomaly descriptions
anomalies = [
    "Have you experienced persistent headaches, especially at the back of your head or neck?",
    "Were there any specific comments or observations about cysts in the ultrasound reports?",
    "Has there been any mention of underdevelopment of the cerebellum in ultrasound reports?",
    # Add more anomaly descriptions as needed
]

# Given line to check
given_line = "I have persistent headaches, especially at the back of my head."

# Function to find the most similar anomaly description
def find_matching_anomaly(line, anomaly_descriptions):
    similarity_scores = [(anomaly, SequenceMatcher(None, line, anomaly).ratio()) for anomaly in anomaly_descriptions]
    most_similar_anomaly = max(similarity_scores, key=lambda x: x[1])
    return most_similar_anomaly

# Find the most similar anomaly description for the given line
matching_anomaly, similarity_score = find_matching_anomaly(given_line, anomalies)

# Print the result
print("Given Line:", given_line)
print("Matching Anomaly:", matching_anomaly)
print("Similarity Score:", similarity_score)
