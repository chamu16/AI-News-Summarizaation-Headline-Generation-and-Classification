from flask import Flask, render_template, request
from summarization import classifyArticle, headlineArticle, summarizeArticle

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    text = ''
    if request.method == 'POST':
        user_input = request.form["article"]
        if request.form["action"] == "sum":
            text = summarizeArticle(user_input)
        elif request.form["action"] == "head":
            text = headlineArticle(user_input)
        elif request.form["action"] == "class":
            text = classifyArticle(user_input)
        else:
            text = "invalid action"
    
    return render_template('index.html', output=text)

if __name__ == "__main__":
    app.run(debug=True)
