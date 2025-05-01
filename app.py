from dataclasses import dataclass
from typing import Iterable

from flask import Flask, render_template, request, redirect
from numpy.random.mtrand import Sequence
from sklearn.feature_extraction.text import TfidfVectorizer
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@dataclass
class TFIDFEntry:
    word: str
    tf: int
    idf: float


@dataclass
class DocumentData:
    filename: str
    tfidf_data: Sequence[TFIDFEntry]


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    files = request.files.getlist('file')
    return render_template(
        'file_page.html',
        files_analysis=list(analyze_files(files))
    )


def analyze_files(files: Sequence[FileStorage]) -> Iterable[DocumentData]:
    vectorizer: TfidfVectorizer = TfidfVectorizer(input='file', norm=None)

    tfidf_matrix = vectorizer.fit_transform(files)
    words = vectorizer.get_feature_names_out()
    idf_values = vectorizer.idf_
    tf_matrix = tfidf_matrix.toarray() / idf_values

    for i, file in enumerate(files):
        filename = secure_filename(file.filename)

        yield DocumentData(
            filename,
            sorted(
                (TFIDFEntry(word, int(tf), round(idf, 3)) for word, tf, idf in zip(words, tf_matrix[i], idf_values) if tf != 0),
                key=lambda x: x.idf,
                reverse=True
            )[:50]
        )


if __name__ == '__main__':
    app.run()
