from flask import Flask, render_template, request, send_from_directory
import joblib
import pandas as pd
import difflib
import re


app = Flask(__name__)


tfv = joblib.load('tfidf_vectorizer.pkl')
sig = joblib.load('sigmoid_kernel.pkl')
indices = pd.read_pickle('anime_indices.pkl')
anime_data = pd.read_csv('data/anime.csv')

def text_cleaning(text):
    text = re.sub(r'&quot;', '', text)
    text = re.sub(r'.hack//', '', text)
    text = re.sub(r'&#039;', '', text)
    text = re.sub(r'A&#039;s', '', text)
    text = re.sub(r'I&#039;', 'I\'', text)
    text = re.sub(r'&amp;', 'and', text)
    
    return text

anime_data['name'] = anime_data['name'].apply(text_cleaning)

def get_recommendations(title, sig=sig, indices=indices):
    try:
        idx = indices[title]
    except KeyError:
        closest_match = difflib.get_close_matches(title, indices.keys(), n=1)
        if closest_match:
            title = closest_match[0]
            idx = indices[title]
        else:
            return ["No match found for the provided anime name."]

    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:11]
    anime_indices = [i[0] for i in sig_scores]

    return list(anime_data['name'].iloc[anime_indices].values)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        book_name = request.form["book_name"]
        recommendations = get_recommendations(book_name)
        return render_template("index.html", recommendations=recommendations)
    return render_template("index.html", recommendations=None)

@app.route('/data-report')
def data_report():
    images = ['top-10-anime-based-on-rating.png',
            'top-10-anime-based-on-audience.png',
            'distribution-of-rating-website-and-user.png',
            'medium-of-streaming.png',]
    return render_template('data_report.html', images=images)

@app.route('/download/<path:filename>', methods=['GET'])
def download_image(filename):
    return send_from_directory('static/images', filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True , port=5000)
