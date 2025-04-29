from transformers import pipeline

sentiment = pipeline("sentiment-analysis")
print(sentiment("I really loved that burger!"))  # Example from your slide
