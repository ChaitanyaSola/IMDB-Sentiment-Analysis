import gradio as gr
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec).max()

    if prediction == 1:
        return f"Positive 😊 (Confidence: {probability:.2f})"
    else:
        return f"Negative 😡 (Confidence: {probability:.2f})"

interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Enter movie review here..."),
    outputs="text",
    title="IMDB Sentiment Analysis",
    description="Model trained on IMDB dataset using TF-IDF + Logistic Regression."
)

interface.launch()
