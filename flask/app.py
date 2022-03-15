from flask import Flask
app = Flask(__name__)
@app.route('/')
# def index():
#     return "안녕하세요. <h1> 홈페이지 입니다.</h>"

def homepage():
    return """
    <h1>Hello world!</h1>

    <iframe src="http://kko.to/-Ck_miyBg" width="1200" height="800" frameborder="0" allowfullscreen></iframe>
    """

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)    