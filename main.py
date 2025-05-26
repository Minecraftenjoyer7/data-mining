import hashlib
import json
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import os
import fitz
from fpdf import FPDF
import numpy as np
import seaborn as sns
from PIL import Image
from flask import flash
import matplotlib.pyplot as plt
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from fpdf import FPDF
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, redirect, url_for, request, jsonify ,session,send_file
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import DataRequired, URL
from flask_ckeditor import CKEditor
from scipy.cluster.hierarchy import fcluster
from flask_gravatar import Gravatar
from flask_login import UserMixin, login_user, LoginManager, current_user, logout_user , login_required
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import relationship, DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Integer, String, Text, ForeignKey, Float
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, r2_score
from sklearn.tree import plot_tree,DecisionTreeClassifier
import matplotlib
matplotlib.use('Agg')
#=======================================================================================================================
app = Flask(__name__,static_folder="./static",template_folder="./templates")
YOUR_DOMAIN = 'http://localhost:4242'
load_dotenv()


app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")
app.config['UPLOAD_FOLDER'] = 'static/assets/datasets'
ckeditor = CKEditor(app)
Bootstrap5(app)

login_manager = LoginManager()
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return db.get_or_404(User, user_id)

class Base(DeclarativeBase):
    pass

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data_mining.db'
db = SQLAlchemy(model_class=Base)
db.init_app(app)


class User(UserMixin,db.Model):
    __tablename__ = "users"
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    password: Mapped[str] = mapped_column(String(200), nullable=False)
    def get_id(self):
           return (self.user_id)
    def to_dict(self):
        dictionary = {}
        for column in self.__table__.columns:
            dictionary[column.name] = getattr(self, column.name)
        return dictionary


with app.app_context():
    db.create_all()

class dataset_form(FlaskForm):
    dataset_name = StringField('dataset name', validators=[DataRequired()])
    dataset_path = FileField("Upload datasets",validators=[FileAllowed(['csv', 'xls', 'xlsx'], 'CSV files'),DataRequired()])
    submit = SubmitField('Done')

#=======================================================================================================================
def get_current_path():
    with open("static/assets/current_path.txt", "r") as file:
        dataset_path = file.read().strip()
    return dataset_path

def get_current_file():
    dataset_path = get_current_path()
    try:
        extension = os.path.splitext(dataset_path)[1]
        if extension == '.csv':
            file = pd.read_csv(dataset_path)
        elif extension in ['.xls', '.xlsx']:
            file = pd.read_excel(dataset_path)
        else:
            return jsonify({"error": {"message": "Unsupported file type."}}), 400
        return file
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500

def get_test_data_file():
    with open("static/assets/current_test_dataset.txt", "r") as file:
        dataset_path = file.read().strip()
    return dataset_path
def get_all_graphs_file_path():
    with open("static/assets/current_all_graphs_path.txt", "r") as file:
        dataset_path = file.read().strip()
    return dataset_path
def get_current_dataset_name():
    with open("static/assets/current_dataset_name.txt", "r") as file:
        dataset_name = file.read().strip()
    return dataset_name
def get_current_model_dataset_path():
    with open("static/assets/current_model_dataset.txt") as file:
        dataset_path = file.read().strip()
    return dataset_path

def get_current_model_dataset():
    dataset_path = get_current_model_dataset_path()
    try:
        extension = os.path.splitext(dataset_path)[1]
        if extension == '.csv':
            file = pd.read_csv(dataset_path)
        elif extension in ['.xls', '.xlsx']:
            file = pd.read_excel(dataset_path)
        else:
            return jsonify({"error": {"message": "Unsupported file type."}}), 400
        return file
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500
def get_current_file_extension():
    dataset_path = get_current_path()
    try:
        extension = os.path.splitext(dataset_path)[1]
        return extension
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500
def save_changes(file):
    dataset_path = get_current_path()
    try:
        extension = get_current_file_extension()
        if extension == '.csv':
            file.to_csv(dataset_path, index=False)
        elif extension in ['.xls', '.xlsx']:
            file.to_excel(dataset_path, index=False)
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500

def has_outliers(df, column):
    if column not in df.columns:
        return jsonify({"error": {"message": "not found"}}), 400

    if not (pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column])):
        return False

    col_data = df[column].dropna()
    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]

    return not outliers.empty

def has_nan(df,col):
    if col == "all":
        return df.isna().values.any()
    else:
        return df[col].isna().values.any()

def has_duplicates(df):
    return df.duplicated().any()


def plot_to_base64():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return image_base64

def get_current_displayed_graph():
    with open("static/assets/current_displayed_graph.txt", "r") as file:
        graph = file.read().strip()
    return graph

def get_current_report():
    with open("static/assets/current_report.txt") as file:
        dataset_path = file.read().strip()
    try:
        extension = os.path.splitext(dataset_path)[1]
        if extension == '.csv':
            file = pd.read_csv(dataset_path)
        else:
            return jsonify({"error": {"message": "Unsupported file type."}}), 400
        return file
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500
def convert_datatype(df, column, datatype):
    current_dtype = df[column].dtype

    if str(current_dtype) == datatype:
        return f"Column is already {datatype}"

    try:
        temp_series = df[column]

        if datatype in ["int64", "float64"]:
            converted = pd.to_numeric(temp_series, errors='coerce')
            failed_count = converted.isna().sum() - temp_series.isna().sum()

            if failed_count > 0:
                return f"Cannot safely convert: {failed_count} values would become NaN."

            df[column] = converted
            if datatype == "int64":
                df[column] = df[column].astype("Int64")

        elif datatype == "object":
            df[column] = temp_series.astype(str)

        elif datatype == "bool":
            if temp_series.dtype == 'object':
                mapped = temp_series.str.lower().map({'true': True, 'false': False})
                failed_count = mapped.isna().sum() - temp_series.isna().sum()

                if failed_count > 0:
                    return f"Cannot safely convert: {failed_count} values couldn't map to bool"
                df[column] = mapped
            df[column] = df[column].astype(bool)

        elif datatype == "datetime64[ns]":
            converted = pd.to_datetime(temp_series, errors='coerce')
            failed_count = converted.isna().sum() - temp_series.isna().sum()

            if failed_count > 0:
                return f"Cannot safely convert: {failed_count} values would become NAN"
            df[column] = converted

        elif datatype == "category":
            df[column] = temp_series.astype('category')

        save_changes(df)
        return f"Successfully converted column '{column}' to {datatype}"

    except Exception as e:
        return f"Error converting column '{column}' to {datatype}: {e}"

def add_normalized_column(df,column,method,replace):
    if replace == False:
        new_col = f"{column}(normalized)"
    else:
        new_col = column

    if method == "Min-Max":
        min_val = df[column].min()
        max_val = df[column].max()
        df[new_col] = (df[column] - min_val) / (max_val - min_val)
    else:
        mean = df[column].mean()
        std = df[column].std()
        df[new_col] = (df[column] - mean) / std

    save_changes(df)

def hierarchical_clustering_method(file,hierarchical_clustering_dataset,method,metric,t):
    cat_cols = []
    num_cols = []
    for col in hierarchical_clustering_dataset.columns:
        if hierarchical_clustering_dataset[col].dtype in ["int64", "float64"]:
            num_cols.append(col)
        else:
            cat_cols.append(col)
    preprocessor = ColumnTransformer(
        [('ohe', OneHotEncoder(drop='first'), cat_cols), ('scale', StandardScaler(), num_cols)])
    X_scaled = preprocessor.fit_transform(hierarchical_clustering_dataset)
    if hasattr(X_scaled, "toarray"):
        X_scaled = X_scaled.toarray()

    Z = linkage(X_scaled, method=method, metric=metric)
    plt.figure(figsize=(20, 20))
    dendrogram(Z)
    plt.title("Dendrogram")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    graph = plot_to_base64()
    with open("static/assets/current_displayed_graph.txt", "w") as f:
        f.write(graph)
    if t != None:
        file['Cluster (Hierarchical)'] = fcluster(Z, t=3, criterion='maxclust')
        save_changes(file)

        sil_score = silhouette_score(X_scaled, file['Cluster (Hierarchical)'])
        db_score = davies_bouldin_score(X_scaled, file['Cluster (Hierarchical)'])

        current_path = get_current_path()
        filename = os.path.splitext(os.path.basename(current_path))[0]
        report_path = os.path.join(os.path.dirname(current_path),f"{filename} (hierarchical report).csv").replace("\\", "/")
        pd.DataFrame({
            'Metric': ['Silhouette Score', 'Davies-Bouldin Index'],
            'Value': [sil_score, db_score]
        }).to_csv(report_path, index=False)
        with open("static/assets/current_report.txt", "w") as f:
            f.write(report_path)
def kmeans_method(k, file, kmeans_dataset):
    cat_cols = []
    num_cols = []
    for col in kmeans_dataset.columns:
        if kmeans_dataset[col].dtype in ["int64", "float64"]:
            num_cols.append(col)
        else:
            cat_cols.append(col)
    preprocessor = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), cat_cols), ('scaler', StandardScaler(), num_cols)])
    X_processed = preprocessor.fit_transform(kmeans_dataset)
    encoded_cols = preprocessor.named_transformers_['encoder'].get_feature_names_out(cat_cols)
    final_columns = list(encoded_cols) + num_cols
    X_scaled = pd.DataFrame(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed, columns=final_columns)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_scaled)
    file['Cluster (Kmeans)'] = kmeans.labels_

    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    db_score = davies_bouldin_score(X_scaled, kmeans.labels_)

    save_changes(file)
    ext = get_current_file_extension()
    current_path = get_current_path()
    base_path = os.path.dirname(current_path)
    filename = os.path.splitext(os.path.basename(current_path))[0]
    data_modeling_path = os.path.join(base_path, f"{filename} (kmeans){ext}").replace("\\", "/")
    if ext in [".xls", ".xlsx"]:
        X_scaled.to_excel(data_modeling_path, index=False)
    else:
        X_scaled.to_csv(data_modeling_path, index=False)
    with open("static/assets/current_model_dataset.txt", "w") as f:
        f.write(data_modeling_path)
    report_path = os.path.join(base_path, f"{filename} (kmeans report).csv").replace("\\", "/")
    pd.DataFrame({
        'Metric': ['Silhouette Score', 'Davies-Bouldin Index'],
        'Value': [sil_score, db_score]
    }).to_csv(report_path, index=False)
    with open("static/assets/current_report.txt", "w") as f:
        f.write(report_path)

def kmedoids_method(k, file, kmedoids_dataset):
    cat_cols = []
    num_cols = []
    for col in kmedoids_dataset.columns:
        if kmedoids_dataset[col].dtype in ["int64", "float64"]:
            num_cols.append(col)
        else:
            cat_cols.append(col)
    preprocessor = ColumnTransformer([('encode', OneHotEncoder(drop='first', sparse_output=False), cat_cols),('scale', StandardScaler(), num_cols)])
    X_scaled = preprocessor.fit_transform(kmedoids_dataset)
    encoded_cols = preprocessor.named_transformers_['encode'].get_feature_names_out(cat_cols)
    final_cols = list(encoded_cols) + num_cols
    X_scaled = pd.DataFrame(X_scaled, columns=final_cols)
    kmedoids = KMedoids(n_clusters=k)
    kmedoids.fit(X_scaled)
    file['Cluster (KMedoids)'] = kmedoids.labels_

    sil_score = silhouette_score(X_scaled, kmedoids.labels_)
    db_score = davies_bouldin_score(X_scaled, kmedoids.labels_)

    save_changes(file)
    ext = get_current_file_extension()
    current_path = get_current_path()
    base_path = os.path.dirname(current_path)
    filename = os.path.splitext(os.path.basename(current_path))[0]
    data_modeling_path = os.path.join(base_path, f"{filename} (kmedoids){ext}").replace("\\", "/")
    if ext in [".xls", ".xlsx"]:
        X_scaled.to_excel(data_modeling_path, index=False)
    else:
        X_scaled.to_csv(data_modeling_path, index=False)
    with open("static/assets/current_model_dataset.txt", "w") as f:
        f.write(data_modeling_path)

    report_path = os.path.join(base_path, f"{filename} (kmedoids_scores).csv").replace("\\", "/")
    pd.DataFrame({
        'Metric': ['Silhouette Score', 'Davies-Bouldin Index'],
        'Value': [sil_score, db_score]
    }).to_csv(report_path, index=False)
    with open("static/assets/current_report.txt", "w") as f:
        f.write(report_path)

def knn_method(k,file,x,y,column_to_predict):
    cat_cols = []
    num_cols = []
    for col in x.columns:
        if x[col].dtype in ["int64", "float64"]:
            num_cols.append(col)
        else:
            cat_cols.append(col)
    preprocessor = ColumnTransformer([('encode', OneHotEncoder(drop='first', sparse_output=False), cat_cols), ('scale', StandardScaler(), num_cols)])
    pipeline = Pipeline([('preprocess', preprocessor), ('knn', KNeighborsClassifier(n_neighbors=k))])
    pipeline.fit(x, y)
    y_pred = pipeline.predict(x)
    report = pd.DataFrame(classification_report(y, y_pred,output_dict=True)).transpose().round(2)
    report.index.name = 'Class/Metric'
    file[f'Predicted_{column_to_predict}(knn)'] = pipeline.predict(x)
    save_changes(file)
    ext = get_current_file_extension()
    current_path = get_current_path()
    base_path = os.path.dirname(current_path)
    filename = os.path.splitext(os.path.basename(current_path))[0]
    data_modeling_path = os.path.join(base_path, f"{filename} (knn report){ext}").replace("\\", "/")
    if ext in [".xls", ".xlsx"]:
        report.to_excel(data_modeling_path, index=True)
    else:
        report.to_csv(data_modeling_path, index=True)
    with open("static/assets/current_model_dataset.txt", "w") as f:
        f.write(data_modeling_path)

def decision_tree_method(max_depth,file,x,y,column_to_predict):
    cat_cols = []
    num_cols = []
    for col in x.columns:
        if x[col].dtype in ["int64", "float64"]:
            num_cols.append(col)
        else:
            cat_cols.append(col)
    preprocessor = ColumnTransformer([('ohe', OneHotEncoder(drop='first'), cat_cols), ('scale', StandardScaler(), num_cols)])
    pipeline = Pipeline([('prep', preprocessor), ('tree', DecisionTreeClassifier(max_depth=max_depth))])
    pipeline.fit(x, y)

    plt.figure(figsize=(20, 10))
    tree_model = pipeline.named_steps['tree']
    preprocessor.fit(x)
    feature_names = preprocessor.get_feature_names_out()
    plot_tree(tree_model, feature_names=feature_names, class_names=True, filled=True, rounded=True)
    graph = plot_to_base64()
    with open("static/assets/current_displayed_graph.txt", "w") as f:
        f.write(graph)
    y_pred = pipeline.predict(x)
    report = pd.DataFrame(classification_report(y, y_pred,output_dict=True)).transpose()
    file[f'Predicted_{column_to_predict}(decision_tree)'] = pipeline.predict(x)
    report.index.name = 'Class/Metric'
    save_changes(file)
    ext = get_current_file_extension()
    current_path = get_current_path()
    base_path = os.path.dirname(current_path)
    filename = os.path.splitext(os.path.basename(current_path))[0]
    data_modeling_path = os.path.join(base_path, f"{filename} (decision_tree report){ext}").replace("\\", "/")
    if ext in [".xls", ".xlsx"]:
        report.to_excel(data_modeling_path, index=True)
    else:
        report.to_csv(data_modeling_path, index=True)
    with open("static/assets/current_model_dataset.txt", "w") as f:
        f.write(data_modeling_path)

def linear_regression_method(file,x,y,column_to_predict):
    cat_cols = []
    num_cols = []
    for col in x.columns:
        if x[col].dtype in ["int64", "float64"]:
            num_cols.append(col)
        else:
            cat_cols.append(col)
    preprocessor = ColumnTransformer([('ohe', OneHotEncoder(drop='first'), cat_cols), ('scale', StandardScaler(), num_cols)])
    pipeline = Pipeline([('prep', preprocessor), ('model', LinearRegression())])
    pipeline.fit(x, y)
    y_pred = pipeline.predict(x)
    file[f'Predicted_{column_to_predict}(linear_regression)'] = pipeline.predict(x)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    report = pd.DataFrame({
        'MSE': [mse],
        'R-Squared': [r2]
    })
    save_changes(file)
    ext = get_current_file_extension()
    current_path = get_current_path()
    base_path = os.path.dirname(current_path)
    filename = os.path.splitext(os.path.basename(current_path))[0]
    data_modeling_path = os.path.join(base_path, f"{filename} (linear_regression report){ext}").replace("\\", "/")
    if ext in [".xls", ".xlsx"]:
        report.to_excel(data_modeling_path, index=False)
    else:
        report.to_csv(data_modeling_path, index=False)
    with open("static/assets/current_model_dataset.txt", "w") as f:
        f.write(data_modeling_path)



#=======================================================================================================================
@app.route("/",methods=["GET","POST"])
def choosing():
    form = dataset_form()
    if form.validate_on_submit():
        dataset_name = form.dataset_name.data
        dataset_file = form.dataset_path.data
        dataset_filename = dataset_file.filename
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        dataset_path_ = os.path.join(app.config['UPLOAD_FOLDER'], dataset_filename).replace("\\", "/")
        dataset_file.save(dataset_path_)
        base_path = os.path.dirname(dataset_path_)
        filename = os.path.splitext(os.path.basename(dataset_path_))[0]
        graphs_path = os.path.join(base_path, f"{filename} (all graphs).csv").replace("\\", "/")
        pd.DataFrame({"graphs": []}).to_csv(graphs_path, index=False)
        with open("static/assets/current_path.txt", "w") as file:
            file.write(dataset_path_)
        with open("static/assets/current_dataset_name.txt", "w") as file:
            file.write(dataset_name)
        with open("static/assets/current_all_graphs_path.txt", "w") as f:
            f.write(graphs_path)
        with open("static/assets/current_pdf_used.txt", "w") as file:
            file.write("")
        with open("static/assets/current_test_dataset.txt", "w") as file:
            file.write("")
        with open("static/assets/current_model_dataset.txt", "w") as file:
            file.write("")
        with open("static/assets/current_displayed_graph.txt", "w") as f:
            f.write("")
        with open("static/assets/current_report.txt", "w") as f:
            f.write("")
        return redirect(url_for('home'))
    return render_template("select.html",form=form)

@app.route("/home", methods=["GET", "POST"])
@app.route("/home/<item_name>", methods=["GET", "POST"])
def home(item_name = "all"):
    with open("static/assets/current_path.txt", "r") as file:
        dataset_path = file.read().strip()
    with open("static/assets/current_dataset_name.txt", "r") as file:
        name = file.read().strip()
    width = "100%"
    context = {"width":width, "name": name}
    hasNan = False
    hasDuplicates = False
    hasOutliers = False
    col_datatype = ""
    if not dataset_path:
        return render_template("index.html", **context)
    try:
        context["size"] = round(os.path.getsize(dataset_path) / (1024 * 1024), 3)
        extension = os.path.splitext(dataset_path)[1]
        context["extension"] = extension
        if extension == '.csv':
            file = pd.read_csv(dataset_path)
        elif extension in ['.xls', '.xlsx']:
            file = pd.read_excel(dataset_path)
        else:
            return jsonify({"error": {"message": "Unsupported file type."}}), 400

        columns = file.columns
        session['dataset_path'] = dataset_path
        session['dataset_name'] = name
        head = file.head(10)
        if len(columns) > 6:
            width = str(len(columns) * 100) + "px"

        if item_name == "all":
            hasNan = has_nan(file, "all")
            hasDuplicates = has_duplicates(file)
        elif item_name:
            hasNan = has_nan(file, item_name)
            hasOutliers = has_outliers(file, item_name)
            col_datatype = file[item_name].dtype
        context.update({
            "table": head,
            "rows": file.shape[0],
            "cols": columns,
            "width":width,
            "hasNan":hasNan,
            "total_nan_values":file.isna().values.sum(),
            "hasDuplicates":hasDuplicates,
            "hasOutliers":hasOutliers,
            "item_name":item_name,
            "col_datatype":col_datatype,
        })
        return render_template("index.html", **context)
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500


@app.route("/home/NAN_table/<col_name>", methods=["GET", "POST"])
def NAN_table(col_name):
    file = get_current_file()
    try:
        if col_name == "all":
            file = file[file.isna().any(axis=1)]
        else:
            file = file[file[col_name].isna()]
        ss = {
            "table": file,
            "cols": file.columns,
            "item_name":col_name,
        }
        return render_template("NAN_values_table.html", **ss)
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500


@app.route("/home/Duplicated_table", methods=["GET", "POST"])
def Duplicates_table():
    file = get_current_file()
    try:
        file = file[file.duplicated(keep=False)]
        ss = {
            "table": file,
            "cols": file.columns
        }
        return render_template("Duplicated_values_table.html", **ss)
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500


@app.route("/home/outlier_graph/<col_name>", methods=["GET", "POST"])
def Outliers_graph(col_name):
    file = get_current_file()
    try:
        file = file[col_name].dropna()
        Q1 = file.quantile(0.25)
        Q3 = file.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = file[(file < lower_bound) | (file > upper_bound)]

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.boxplot(file, vert=False)
        ax.set_title(f'Outliers in "{col_name}"')
        image = plot_to_base64()
        with open("static/assets/current_displayed_graph.txt", "w") as f:
            f.write(image)
        ss = {
            "image" : image,
            "col_name" : col_name,
            "outliers": outliers,
        }
        return render_template("outliers_graph.html", **ss)
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500


@app.route("/home/data_cleaning/<col_name>", methods=["GET", "POST"])
def data_cleaning(col_name):
    alert = ""
    file = get_current_file()
    try:
        if col_name == "all":
            nan_action = request.form.get('nan_action')
            nan_value = request.form.get('nan_value')
            duplicate_action = request.form.get('duplicate_action')
            first_column = request.form.get('first_column')
            if nan_action == "replace":
                file = file.fillna(nan_value)
            elif nan_action == "remove":
                file = file.dropna()

            if duplicate_action == "remove_duplicates":
                file = file.drop_duplicates()

            if first_column == "remove":
                if file.shape[1] > 0:
                    file = file.drop(columns=[file.columns[0]])
            save_changes(file)
        else:
            nan_action = request.form.get('nan_action')
            nan_value = request.form.get('nan_value')
            outliers_action = request.form.get('outliers_action')
            datatype_convertion = request.form.get('datatype_value')
            add_normalized_col = request.form.get('add_normalized_col')
            replace_normalized = request.form.get('replace_normalized') == 'replace'
            if nan_action == "replace":
                file[col_name] = file[col_name].fillna(nan_value)
            elif nan_action == "remove":
                file = file[file[col_name].notna()]
            elif nan_action == "mean":
                file[col_name] = file[col_name].fillna(round(file[col_name].mean()))

            if outliers_action == "remove_outliers":
                if pd.api.types.is_numeric_dtype(file[col_name]) and file[col_name].dtype != bool:
                    col_data = file[col_name].dropna()
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    file = file[(file[col_name] >= lower_bound) & (file[col_name] <= upper_bound)]

            if datatype_convertion not in ["keep",None]:
                alert = convert_datatype(file,col_name,datatype_convertion)

            if add_normalized_col not in [None,"none"]:
                add_normalized_column(file,col_name,add_normalized_col,replace_normalized)

            save_changes(file)
            if alert != "":
                flash(alert)
        return redirect(url_for('home'))
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500


@app.route("/home/mining_technique_option", methods=["GET", "POST"])
def mining_technique_option():
    with open("static/assets/current_model_dataset.txt", "w") as f:
        f.write("")
    with open("static/assets/current_displayed_graph.txt", "w") as f:
        f.write("")
    return render_template("mining_technique_options.html")

@app.route("/home/replace_values/<col_name>", methods=["GET", "POST"])
def replace_values(col_name):
    file = get_current_file()
    if request.method == "POST":
        old_value = request.form.get("old_value")
        new_value = request.form.get("new_value")
        if old_value is not None and new_value is not None:
            column = file[col_name]
            file[col_name] = column.where(column.isna(),column.astype(str).str.replace(old_value, new_value, regex=False))
        save_changes(file)

        return redirect(url_for('replace_values', col_name=col_name))

    first20 = file[col_name].head(20)
    ss = {
        "col_name": col_name,
        "first_20_rows":first20,
    }
    return render_template("replace.html",**ss)


@app.route("/home/delete_column/<col_name>", methods=["GET", "POST"])
def delete_column(col_name):
    file = get_current_file()
    file = file.drop(col_name, axis=1)
    save_changes(file)
    return redirect(url_for('home'))

@app.route("/home/data_split", methods=["GET", "POST"])
def data_split():
    file = get_current_file()
    ext = get_current_file_extension()
    if request.method == "POST":
        training_percentage = int(request.form.get("range_value"))/100
        test_percentage = 1 - training_percentage
        current_path = get_current_path()
        train_df, test_df = train_test_split(file, test_size=test_percentage, random_state=7)
        base_path = os.path.dirname(current_path)
        filename = os.path.splitext(os.path.basename(current_path))[0]
        if ext in [".xls", ".xlsx"]:
            train_file = os.path.join(base_path, f"{filename} (training data).xlsx").replace("\\", "/")
            test_file = os.path.join(base_path, f"{filename} (test data).xlsx").replace("\\", "/")
        else:
            train_file = os.path.join(base_path, f"{filename} (training data).csv").replace("\\", "/")
            test_file = os.path.join(base_path, f"{filename} (test data).csv").replace("\\", "/")

        try:
            if ext in [".xls", ".xlsx"]:
                train_df.to_excel(train_file, index=False)
                test_df.to_excel(test_file, index=False)
            else:
                train_df.to_csv(train_file, index=False)
                test_df.to_csv(test_file, index=False)

            with open("static/assets/current_path.txt", "w") as file:
                file.write(train_file)
            with open("static/assets/current_test_dataset.txt", "w") as file:
                file.write(test_file)
            flash("Data split successfully")
        except Exception as e:
            flash(f"Error while splitting data: {e}")
        return redirect("data_split")
    return render_template("data_split.html")

@app.route("/home/mining_technique_option/choose_method/<method>",methods=["GET","POST"])
def choose_method(method):
    with open("static/assets/current_model_dataset.txt", "w") as file:
        file.write("")
    with open("static/assets/current_displayed_graph.txt", "w") as file:
        file.write("")
    methods = ['kmeans','kmedoids','hierarchical_clustering','knn','decision_tree','linear_regression','fuzzy']
    if method in methods:
        return redirect(url_for(method))
    else:
        return jsonify({"error": {"message": "route doesn't exist"}}), 500

@app.route("/home/mining_technique_option/hierarchical_clustering",methods=["GET","POST"])
def hierarchical_clustering():
    file = get_current_file()
    hasNan = file.isna().values.any()
    graph = ""
    graph_validation = False
    if "Cluster (Hierarchical)" in file.columns:
        graph_validation = True
    with open("static/assets/current_displayed_graph.txt", "r") as f:
        graph = f.read().strip()

    ss = {
        "hasNan": hasNan,
        "columns": file.columns,
        "graph":graph,
        "graph_validation":graph_validation
    }
    if request.method == "POST":
        method = request.form.get("method")
        metric = request.form.get("metric")
        t = request.form.get("threshold_value")
        columns_to_drop = request.form.getlist("checkbox")
        if columns_to_drop != []:
            hierarchical_clustering_dataset = file.drop(columns_to_drop,axis=1)
        else:
            hierarchical_clustering_dataset = file
        hierarchical_clustering_method(file, hierarchical_clustering_dataset, method, metric,t)
        return redirect("hierarchical_clustering")
    return render_template("hierarchical_clustering.html",**ss)

@app.route("/home/mining_technique_option/kmeans",methods=["GET","POST"])
def kmeans():
    file = get_current_file()
    hasNan = file.isna().values.any()
    graph_validation = False
    if "Cluster (Kmeans)" in file.columns:
        graph_validation = True
    with open("static/assets/current_model_dataset.txt", "r") as f:
        test_dataset_path = f.read().strip()
        extension = os.path.splitext(test_dataset_path)[1]
        if extension == "":
            kmeans_dataset = ""
        elif extension == ".csv":
            kmeans_dataset = pd.read_csv(test_dataset_path)
        else:
            kmeans_dataset = pd.read_excel(test_dataset_path)

    if hasattr(kmeans_dataset, 'columns') and not kmeans_dataset.empty:
        columns = kmeans_dataset.columns
        sample_size = min(10, len(kmeans_dataset))
    else:
        columns = []
        sample_size = 0

    if type(columns) != list:
        columns = columns.to_list()
    ss = {
        "kmeans_dataset":kmeans_dataset,
        "hasNan":hasNan,
        "columns":file.columns,
        "kmeans_dataset_columns":columns,
        "sample_size":sample_size,
        "graph": graph_validation,
    }
    if request.method == "POST":
        k = int(request.form.get("k"))
        columns_to_drop = request.form.getlist("checkbox")
        if columns_to_drop != []:
            kmeans_dataset = file.drop(columns_to_drop,axis=1)
        else:
            kmeans_dataset = file
        kmeans_method(k,file,kmeans_dataset)
        return redirect("kmeans")
    return render_template("kmeans.html", **ss)

@app.route("/home/mining_technique_option/kmedoids",methods=["GET","POST"])
def kmedoids():
    file = get_current_file()
    hasNan = file.isna().values.any()
    graph_validation = False
    if "Cluster (KMedoids)" in file.columns:
        graph_validation = True
    with open("static/assets/current_displayed_graph.txt", "r") as f:
        graph = f.read().strip()
    with open("static/assets/current_model_dataset.txt", "r") as f:
        test_dataset_path = f.read().strip()
        extension = os.path.splitext(test_dataset_path)[1]
        if extension == "":
            kmedoids_dataset = ""
        elif extension == ".csv":
            kmedoids_dataset = pd.read_csv(test_dataset_path)
        else:
            kmedoids_dataset = pd.read_excel(test_dataset_path)

    if hasattr(kmedoids_dataset, 'columns') and not kmedoids_dataset.empty:
        columns = kmedoids_dataset.columns
        sample_size = min(10, len(kmedoids_dataset))
    else:
        columns = []
        sample_size = 0

    if type(columns) != list:
        columns = columns.to_list()
    ss = {
        "kmedoids_dataset":kmedoids_dataset,
        "hasNan":hasNan,
        "columns":file.columns,
        "kmedoids_dataset_columns":columns,
        "sample_size":sample_size,
        "graph": graph_validation,
    }
    if request.method == "POST":
        k = int(request.form.get("k"))
        columns_to_drop = request.form.getlist("checkbox")
        if columns_to_drop != []:
            kmedoids_dataset = file.drop(columns_to_drop,axis=1)
        else:
            kmedoids_dataset = file
        kmedoids_method(k,file,kmedoids_dataset)
        return redirect("kmedoids")
    return render_template("kmedoids.html",**ss)

@app.route("/home/mining_technique_option/knn",methods=["GET","POST"])
def knn():
    file = get_current_file()
    prediction = request.args.get("prediction")
    hasNan = file.isna().values.any()
    graph_validation = False
    if prediction is not None and f'Predicted_{prediction}(knn)' in file.columns:
        graph_validation = True
    with open("static/assets/current_model_dataset.txt", "r") as f:
        test_dataset_path = f.read().strip()
        extension = os.path.splitext(test_dataset_path)[1]
        if extension == "":
            knn_report = ""
        elif extension == ".csv":
            knn_report = pd.read_csv(test_dataset_path)
        else:
            knn_report = pd.read_excel(test_dataset_path)

    if hasattr(knn_report, 'columns') and not knn_report.empty:
        columns = knn_report.columns
        sample_size = min(6, len(knn_report))
    else:
        columns = []
        sample_size = 0

    if type(columns) != list:
        columns = columns.to_list()
    s = 20 if len(file) > 19 else len(file)
    ss = {
        "file":file.head(s),
        "knn_report": knn_report,
        "hasNan": hasNan,
        "columns": file.columns,
        "knn_dataset_columns": columns,
        "sample_size": sample_size,
        "prediction": prediction,
        "graph": graph_validation,
    }
    if request.method == "POST":
        k = int(request.form.get("k"))
        columns_to_drop = request.form.getlist("checkbox")
        column_to_predict = request.form.get("column_to_predict")
        if not column_to_predict in columns_to_drop:
            columns_to_drop.append(column_to_predict)
        x = file.drop(columns_to_drop, axis=1)
        y = file[column_to_predict]
        knn_method(k, file, x, y, column_to_predict)
        return redirect(url_for("knn", prediction=column_to_predict))
    return render_template("knn.html", **ss)

@app.route("/home/mining_technique_option/decision_tree",methods=["GET","POST"])
def decision_tree():
    file = get_current_file()
    prediction = request.args.get("prediction")
    hasNan = file.isna().values.any()
    graph = ""
    with open("static/assets/current_displayed_graph.txt", "r") as f:
        graph = f.read().strip()
    with open("static/assets/current_model_dataset.txt", "r") as f:
        test_dataset_path = f.read().strip()
        extension = os.path.splitext(test_dataset_path)[1]
        if extension == "":
            decision_tree_report = ""
        elif extension == ".csv":
            decision_tree_report = pd.read_csv(test_dataset_path)
        else:
            decision_tree_report = pd.read_excel(test_dataset_path)

    if hasattr(decision_tree_report, 'columns') and not decision_tree_report.empty:
        columns = decision_tree_report.columns
        sample_size = min(6, len(decision_tree_report))
    else:
        columns = []
        sample_size = 0

    if type(columns) != list:
        columns = columns.to_list()
    s = 20 if len(file) > 19 else len(file)
    ss = {
        "file": file.head(s),
        "decision_tree_report": decision_tree_report,
        "hasNan": hasNan,
        "columns": file.columns,
        "decision_tree_dataset_columns": columns,
        "sample_size": sample_size,
        "prediction": prediction,
        "graph": graph,
    }
    if request.method == "POST":
        max_depth = int(request.form.get("max_depth"))
        column_to_predict = request.form.get("column_to_predict")
        columns_to_drop = request.form.getlist("checkbox")
        if not column_to_predict in columns_to_drop:
            columns_to_drop.append(column_to_predict)
        x = file.drop(columns_to_drop, axis=1)
        y = file[column_to_predict]
        decision_tree_method(max_depth, file, x, y, column_to_predict)
        return redirect(url_for("decision_tree", prediction=column_to_predict))
    return render_template("decision_tree.html", **ss)

@app.route("/home/mining_technique_option/linear_regression",methods=["GET","POST"])
def linear_regression():
    file = get_current_file()
    prediction = request.args.get("prediction")
    hasNan = file.isna().values.any()
    columns_to_predict = []
    mse = ""
    r2 = ""
    graph_validation = False
    if prediction is not None and f'Predicted_{prediction}(linear_regression)' in file.columns:
        graph_validation = True
    with open("static/assets/current_model_dataset.txt", "r") as f:
        test_dataset_path = f.read().strip()
        extension = os.path.splitext(test_dataset_path)[1]
        if extension == "":
            linear_regression_report = ""
        elif extension == ".csv":
            linear_regression_report = pd.read_csv(test_dataset_path)
            mse = linear_regression_report["MSE"][0]
            r2 = linear_regression_report["R-Squared"][0]
        else:
            linear_regression_report = pd.read_excel(test_dataset_path)
            mse = linear_regression_report["MSE"][0]
            r2 = linear_regression_report["R-Squared"][0]

    if hasattr(linear_regression_report, 'columns') and not linear_regression_report.empty:
        columns = linear_regression_report.columns
        sample_size = min(1, len(linear_regression_report))
    else:
        columns = []
        sample_size = 0
    if type(columns) != list:
        columns = columns.to_list()
    s = 20 if len(file) > 19 else len(file)
    columns_to_predict = [col for col in file.columns if file[col].dtype in ['int64', 'float64']]
    ss = {
        "file": file.head(s),
        "r2":r2,
        "mse":mse,
        "hasNan": hasNan,
        "columns": file.columns,
        "columns_to_predict":columns_to_predict,
        "sample_size": sample_size,
        "prediction": prediction,
        "graph": graph_validation,
    }
    if request.method == "POST":
        column_to_predict = request.form.get("column_to_predict")
        columns_to_drop = request.form.getlist("checkbox")
        if not column_to_predict in columns_to_drop:
            columns_to_drop.append(column_to_predict)
        x = file.drop(columns_to_drop, axis=1)
        y = file[column_to_predict]
        linear_regression_method(file,x,y,column_to_predict)
        return redirect(url_for("linear_regression", prediction=column_to_predict))
    return render_template("linear_regression.html", **ss)

# @app.route("/home/mining_technique_option/fuzzy", methods=["GET", "POST"])
# def fuzzy():
#     file = get_current_file()
#     numeric_columns = [col for col in file.columns if file[col].dtype in ['int64', 'float64']]
#     ss = {"columns": numeric_columns}
#
#     if request.method == "POST":
#         input_columns = request.form.getlist("input_columns")
#         output_column = request.form.get("output_column")
#         defuzz_method = request.form.get("defuzz_method", "centroid")  # Default to centroid
#
#         # Validate inputs
#         if not input_columns or not output_column:
#             return render_template("fuzzy.html", columns=file.columns, error="Missing inputs or output")
#         if len(input_columns) < 2:
#             return render_template("fuzzy.html", columns=file.columns, error="At least 2 input columns required")
#         if output_column in input_columns:
#             return render_template("fuzzy.html", columns=file.columns, error="Output column cannot be an input column")
#
#         # Calculate metrics (min, avg, max)
#         metrics = []
#         for col in input_columns + [output_column]:
#             metrics.append({
#                 "column": col,
#                 "min": float(file[col].min()),
#                 "max": float(file[col].max()),
#                 "avg": float(file[col].mean())
#             })
#         ss["metrics"] = metrics
#
#         try:
#             antecedents = {}
#             for col in input_columns:
#                 min_val = file[col].min()
#                 max_val = file[col].max()
#                 universe = np.arange(min_val, max_val + 0.1, (max_val - min_val) / 100)   #np.arange(0, 10.1, 0.1)  -->  [0.0, 0.1, 0.2, ..., 10.0]   -> So you get 101 points perfect for drawing smooth fuzzy curves
#                 antecedents[col] = ctrl.Antecedent(universe = universe, label = col)
#                 antecedents[col].automf(3, names=['poor', 'average', 'good'])
#
#             out_min = file[output_column].min()
#             out_max = file[output_column].max()
#             out_universe = np.arange(out_min, out_max + 0.1, (out_max - out_min) / 100)
#             consequent = ctrl.Consequent(universe = out_universe, label = output_column, defuzzify_method=defuzz_method)
#             consequent['low'] = fuzz.trimf(out_universe, [out_min, out_min, (out_min + out_max) / 2])
#             consequent['medium'] = fuzz.trimf(out_universe, [out_min, (out_min + out_max) / 2, out_max])
#             consequent['high'] = fuzz.trimf(out_universe, [(out_min + out_max) / 2, out_max, out_max])
#
#             # Dynamic rules (inspired by university example)
#             rules = []
#             # Rule 1: If any input is poor, output is low
#             poor_conditions = [antecedents[col]['poor'] for col in input_columns]
#             if poor_conditions:
#                 rules.append(ctrl.Rule(poor_conditions[0] | (poor_conditions[1] if len(poor_conditions) > 1 else poor_conditions[0]), consequent['low']))
#             # Rule 2: If any input is average, output is medium
#             avg_conditions = [antecedents[col]['average'] for col in input_columns]
#             if avg_conditions:
#                 rules.append(ctrl.Rule(avg_conditions[0], consequent['medium']))
#             # Rule 3: If any input is good, output is high
#             good_conditions = [antecedents[col]['good'] for col in input_columns]
#             if good_conditions:
#                 rules.append(ctrl.Rule(good_conditions[0] | (good_conditions[1] if len(good_conditions) > 1 else good_conditions[0]), consequent['high']))
#
#             # Create control system
#             fuzzy_system = ctrl.ControlSystem(rules)
#             simulation = ctrl.ControlSystemSimulation(fuzzy_system, clip_to_bounds=True)
#
#             # Example input (using mean values for demo)
#             for col in input_columns:
#                 simulation.input[col] = file[col].mean()
#             simulation.compute()
#             predicted_output = simulation.output[output_column]
#
#             # Save membership function plots
#             plot_dir = "static/plots"
#             os.makedirs(plot_dir, exist_ok=True)
#             plot_paths = []
#             for col in input_columns:
#                 fig, ax = plt.subplots()
#                 for term in ['poor', 'average', 'good']:
#                     ax.plot(antecedents[col].universe, antecedents[col][term].mf, label=term)
#                 ax.set_title(f"{col} Membership")
#                 ax.legend()
#                 plot_path = os.path.join(plot_dir, f"{col}_membership.png")
#                 fig.savefig(plot_path)
#                 plt.close(fig)
#                 plot_paths.append(f"plots/{col}_membership.png")
#             # Consequent plot
#             fig, ax = plt.subplots()
#             for term in ['low', 'medium', 'high']:
#                 ax.plot(consequent.universe, consequent[term].mf, label=term)
#             ax.set_title(f"{output_column} Membership")
#             ax.legend()
#             consequent_plot = os.path.join(plot_dir, f"{output_column}_membership.png")
#             fig.savefig(consequent_plot)
#             plt.close(fig)
#             plot_paths.append(f"plots/{output_column}_membership.png")
#
#             ss.update({
#                 "predicted_output": round(predicted_output, 2),
#                 "plot_paths": plot_paths,
#                 "defuzz_method": defuzz_method
#             })
#
#         except Exception as e:
#             ss["error"] = f"Fuzzy logic error: {str(e)}"
#
#     # Available defuzzification methods
#     ss["defuzz_methods"] = ["centroid", "bisector", "mom", "som", "lom"]
#     return render_template("fuzzy.html", **ss)


@app.route("/home/mining_technique_option/fuzzy", methods=["GET", "POST"])
def fuzzy():
    file = get_current_file()
    numeric_columns = [col for col in file.columns if file[col].dtype in ['int64', 'float64']]
    ss = {
        "columns": numeric_columns,
    }

    if request.method == "POST":
        input_columns = request.form.getlist("input_columns")
        output_column = request.form.get("output_column")
        defuzz_method = request.form.get("defuzz_method", "centroid")

        if not input_columns or not output_column:
            return render_template("fuzzy.html", **ss, error="Missing inputs or output")
        if len(input_columns) < 2:
            return render_template("fuzzy.html", **ss, error="At least 2 input columns required")
        if output_column in input_columns:
            return render_template("fuzzy.html", **ss, error="Output column cannot be an input column")

        metrics = []
        for col in input_columns + [output_column]:
            metrics.append({
                "column": col,
                "min": float(file[col].min()),
                "max": float(file[col].max()),
                "avg": float(file[col].mean())
            })
        try:
            antecedents = {}
            for col in input_columns:
                min_val, max_val = file[col].min(), file[col].max()
                universe = np.arange(min_val, max_val + 0.1, (max_val - min_val) / 100)
                antecedents[col] = ctrl.Antecedent(universe = universe, label = col)
                antecedents[col].automf(3, names=['poor', 'average', 'good'])

            out_min, out_max = file[output_column].min(), file[output_column].max()
            out_universe = np.arange(out_min, out_max + 0.1, (out_max - out_min) / 100)
            consequent = ctrl.Consequent(out_universe, output_column, defuzzify_method=defuzz_method)
            consequent['low'] = fuzz.trimf(out_universe, [out_min, out_min, (out_min + out_max) / 2])
            consequent['medium'] = fuzz.trimf(out_universe, [out_min, (out_min + out_max) / 2, out_max])
            consequent['high'] = fuzz.trimf(out_universe, [(out_min + out_max) / 2, out_max, out_max])

            #=========================================================
            session['input_columns'] = input_columns
            session['output_column'] = output_column
            session['defuzz_method'] = defuzz_method
            session['antecedents'] = {col: {'universe': antecedents[col].universe.tolist(),
                                            'terms': ['poor', 'average', 'good']}for col in input_columns}
            session['consequent'] = {
                'universe': out_universe.tolist(),
                'terms': ['low', 'medium', 'high'],
                'defuzz_method': defuzz_method
            }
            session['metrics'] = metrics

            ss.update({
                "metrics": metrics,
                "input_columns": input_columns,
                "output_column": output_column,
                "defuzz_method": defuzz_method,
                "show_rule_form": True
            })

        except Exception as e:
            return jsonify({"error": {"message": str(e)}}), 500
    return render_template("fuzzy.html", **ss)

@app.route("/home/mining_technique_option/fuzzy_rules", methods=["POST"])
def fuzzy_rules():
    file = get_current_file()
    numeric_columns = [col for col in file.columns if file[col].dtype in ['int64', 'float64']]
    antecedents_data = session.get('antecedents', {})
    consequent_data = session.get('consequent', {})
    input_cols = session.get('input_columns', [])
    output_column = session.get('output_column', '')
    defuzz_method = session.get('defuzz_method', 'centroid')
    input_columns = [col for col in input_cols if col in antecedents_data]

    ss = {
        "columns": numeric_columns,
        "metrics": session.get('metrics', []),
        "input_columns": input_columns,
        "output_column":output_column,
        "defuzz_method": defuzz_method
    }
    print(input_columns)
    antecedents = {}
    for col in input_columns:
        universe = np.array(antecedents_data[col]['universe'])
        antecedents[col] = ctrl.Antecedent(universe, col)
        antecedents[col].automf(3, names=['poor', 'average', 'good'])

    out_universe = np.array(consequent_data['universe'])
    consequent = ctrl.Consequent(out_universe, output_column, defuzzify_method=defuzz_method)
    consequent['low'] = fuzz.trimf(out_universe, [out_universe[0], out_universe[0], (out_universe[0] + out_universe[-1]) / 2])
    consequent['medium'] = fuzz.trimf(out_universe, [out_universe[0], (out_universe[0] + out_universe[-1]) / 2, out_universe[-1]])
    consequent['high'] = fuzz.trimf(out_universe, [(out_universe[0] + out_universe[-1]) / 2, out_universe[-1], out_universe[-1]])

    raw_rules = request.form.getlist('rule_conditions[]')
    rules_data = []
    print(antecedents)
    for item in raw_rules:
        rules_data.extend(item.split(';'))
    print(rules_data)
    if len(rules_data) < 2:
        return render_template("fuzzy.html", **ss, error="At least 2 rules are required")
    session["rules"] = rules_data
    rules = []
    for rule_str in rules_data:
        parts = rule_str.split(',')
        condition = None
        i = 0
        while i < len(parts):
            col = parts[i]
            term = parts[i + 1]
            op = parts[i + 2]

            if op == '=':
                consequent_term = parts[i + 3]
                if condition is None:
                    condition = antecedents[col][term]
                rule = ctrl.Rule(condition, consequent[consequent_term])
                rules.append(rule)
                break

            elif op == '&':
                if condition is None:
                    condition = antecedents[col][term] & antecedents[parts[i + 3]][parts[i + 4]]
                else:
                    condition = condition & antecedents[parts[i + 3]][parts[i + 4]]
                i += 3

            elif op == '|':
                if condition is None:
                    condition = antecedents[col][term] | antecedents[parts[i + 3]][parts[i + 4]]
                else:
                    condition = condition | antecedents[parts[i + 3]][parts[i + 4]]
                i += 3
            else:
                i += 1
    #=======================================
    print(rules)
    clean_rules = []
    for rule in rules:
        rule_str = str(rule)
        first_line = rule_str.split('\n')[0]
        clean_rules.append(first_line)
    print(clean_rules)

    #=======================================
    fuzzy_system = ctrl.ControlSystem(rules)
    print(fuzzy_system)
    simulation = ctrl.ControlSystemSimulation(fuzzy_system, clip_to_bounds=True)
    try:
        print("Simulation input state -->", simulation.input)
        print("Simulation output state -->", dir(simulation.output))
        file.columns = file.columns.str.strip().str.lower()
        valid_columns = set(file.columns)

        used_columns = set()
        for rule_str in rules_data:
            parts = rule_str.split(',')
            i = 0
            while i < len(parts):
                col_candidate = parts[i].strip().lower()
                if col_candidate in valid_columns:
                    used_columns.add(col_candidate)

                op = parts[i + 2].strip().lower() if i + 2 < len(parts) else ''
                if op in ['&', '|'] and i + 3 < len(parts):
                    col_candidate2 = parts[i + 3].strip().lower()
                    if col_candidate2 in valid_columns:
                        used_columns.add(col_candidate2)
                    i += 3
                else:
                    i += 1

        normalized_input_columns = list(used_columns)
        print(antecedents)
        antecedents = {k: v for k, v in antecedents.items() if k in normalized_input_columns}

        print("File columns:", file.columns.tolist())
        print("Input columns:", normalized_input_columns)
        print("antecedents",antecedents)
        print("Filtered input columns from rules:", normalized_input_columns)

        for col in normalized_input_columns:
            # if col not in file.columns:
            #     raise ValueError(f"Input column '{col}' is missing in the uploaded dataset")
            simulation.input[col] = file[col].mean()
        simulation.compute()
        predicted_output = simulation.output[output_column]
        print("s -->  ",simulation)
        plot_dir = "static/plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_paths = []
        for col in normalized_input_columns:
            fig, ax = plt.subplots()
            for term in ['poor', 'average', 'good']:
                ax.plot(antecedents[col].universe, antecedents[col][term].mf, label=term)
            ax.set_title(f"{col} Membership")
            ax.legend()
            plot_path = os.path.join(plot_dir, f"{col}_membership.png")
            fig.savefig(plot_path)
            plt.close(fig)
            plot_paths.append(f"plots/{col}_membership.png")
        fig, ax = plt.subplots()
        for term in ['low', 'medium', 'high']:
            ax.plot(consequent.universe, consequent[term].mf, label=term)
        ax.set_title(f"{output_column} Membership")
        ax.legend()
        consequent_plot = os.path.join(plot_dir, f"{output_column}_membership.png")
        fig.savefig(consequent_plot)
        plt.close(fig)
        plot_paths.append(f"plots/{output_column}_membership.png")
        print(plot_paths)
        print(predicted_output)
        print(defuzz_method)
        print(clean_rules)
        session["input_columns"] = normalized_input_columns
        print("llllll",antecedents)
        ss.update({
            "predicted_output": round(predicted_output, 2),
            "plot_paths": plot_paths,
            "defuzz_method": defuzz_method,
            "clean_rules": clean_rules,
            "show_nav_links":True
        })
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500
    return render_template("fuzzy.html", **ss)

@app.route("/home/mining_technique_option/fuzzy_graph", methods=["GET", "POST"])
def fuzzy_graph():
    file = get_current_file()
    numeric_columns = [col for col in file.columns if file[col].dtype in ['int64', 'float64']]
    metrics = session.get('metrics', [])
    input_columns = session.get('input_columns', [])
    ss = {
        "columns": numeric_columns,
        "input_columns": input_columns,
        "metrics": metrics,
        "output_column": session.get('output_column', ''),
        "defuzz_method": session.get('defuzz_method', 'centroid'),
        "show_nav_links": bool(session.get('rules') and len(session['rules']) > 0),
        "graph": None,
        "result": None,
        "user_inputs": {}
    }

    required_session_keys = ['input_columns', 'metrics', 'antecedents', 'consequent', 'output_column', 'defuzz_method', 'rules']
    if not all(key in session for key in required_session_keys):
        return jsonify({"error": {"message": "Fuzzy system not initialized. Please complete fuzzy setup first."}}), 500
    ant = session["antecedents"]
    antecedents = {}
    for label, info in ant.items():
        universe = np.array(info['universe'])
        antecedents[label] = ctrl.Antecedent(universe, label)
        antecedents[label].automf(3,names=['poor', 'average', 'good'])

    antecedents = {k: v for k, v in antecedents.items() if k in input_columns}
    c_info = session['consequent']
    c_universe = np.array(c_info['universe'])
    consequent = ctrl.Consequent(c_universe, session['output_column'], defuzzify_method=c_info['defuzz_method'])
    consequent['low'] = fuzz.trimf(c_universe, [c_universe[0], c_universe[0], (c_universe[0] + c_universe[-1]) / 2])
    consequent['medium'] = fuzz.trimf(c_universe, [c_universe[0], (c_universe[0] + c_universe[-1]) / 2, c_universe[-1]])
    consequent['high'] = fuzz.trimf(c_universe, [(c_universe[0] + c_universe[-1]) / 2, c_universe[-1], c_universe[-1]])
    rules_data = session['rules']
    rules = []
    for rule_str in rules_data:
        parts = rule_str.split(',')
        condition = None
        i = 0
        while i < len(parts):
            col = parts[i]
            term = parts[i + 1]
            op = parts[i + 2]

            if op == '=':
                consequent_term = parts[i + 3]
                if condition is None:
                    condition = antecedents[col][term]
                rule = ctrl.Rule(condition, consequent[consequent_term])
                rules.append(rule)
                break

            elif op == '&':
                if condition is None:
                    condition = antecedents[col][term] & antecedents[parts[i + 3]][parts[i + 4]]
                else:
                    condition = condition & antecedents[parts[i + 3]][parts[i + 4]]
                i += 3

            elif op == '|':
                if condition is None:
                    condition = antecedents[col][term] | antecedents[parts[i + 3]][parts[i + 4]]
                else:
                    condition = condition | antecedents[parts[i + 3]][parts[i + 4]]
                i += 3
            else:
                i += 1
    if not rules:
        return jsonify({"error": {"message": "No valid rules found. Please define rules in the fuzzy rules page."}}), 500
    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system, clip_to_bounds=True)

    if request.method == "POST":
        for label in antecedents:
            val = float(request.form.get(label, file[label].mean()))
            metric = next((m for m in metrics if m['column'] == label), None)
            if metric and not (metric['min'] <= val <= metric['max']):
                return render_template("fuzzy.html", **ss, error=f"Input for {label} must be between {metric['min']} and {metric['max']}")
            sim.input[label] = val
            ss["user_inputs"][label] = val

        sim.compute()
        ss["result"] = sim.output[session['output_column']]

        fig, ax = plt.subplots(figsize=(8, 4))
        consequent.view(sim=sim, ax=ax)
        ax.set_title(f"{session['output_column']} Membership Function")
        ax.set_xlabel(session['output_column'])
        ax.set_ylabel('Membership Degree')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
        buf.seek(0)
        ss["graph"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close(fig)

    ss["antecedents"] = session['input_columns']

    return render_template("fuzzy_graph.html", **ss)

# def fuzzy_graph():
#     if "antecedents" not in session or "consequent" not in session:
#         return "Fuzzy system not initialized. Please go to the fuzzy setup page first."
#
#     # Load antecedents from session
#     antecedents = {}
#     for label, info in session['antecedents'].items():
#         universe = np.array(info['universe'])
#         antecedents[label] = ctrl.Antecedent(universe, label)
#         antecedents[label].automf(3, names=info['terms'])
#
#     # Load consequent
#     c_info = session['consequent']
#     c_universe = np.array(c_info['universe'])
#     consequent = ctrl.Consequent(c_universe, session['output_column'], defuzzify_method=c_info['defuzz_method'])
#
#     # Define membership functions
#     consequent['low'] = fuzz.trimf(c_universe, [c_universe[0], c_universe[0], (c_universe[-1] + c_universe[0]) / 2])
#     consequent['medium'] = fuzz.trimf(c_universe, [c_universe[0], (c_universe[-1] + c_universe[0]) / 2, c_universe[-1]])
#     consequent['high'] = fuzz.trimf(c_universe, [(c_universe[-1] + c_universe[0]) / 2, c_universe[-1], c_universe[-1]])
#
#     # Dummy rules for demonstration
#     rules = []
#     for ant in antecedents.values():
#         for term in ant.terms:
#             rules.append(ctrl.Rule(ant[term], consequent['medium']))
#     system = ctrl.ControlSystem(rules)
#     sim = ctrl.ControlSystemSimulation(system)
#
#     graph = None
#     result = None
#     metrics = []
#
#     # Prepare stats
#     df = pd.DataFrame(session['df'])
#     for column in antecedents.keys():
#         series = pd.to_numeric(df[column], errors='coerce')
#         metrics.append({
#             'column': column,
#             'min': round(series.min(), 2),
#             'avg': round(series.mean(), 2),
#             'max': round(series.max(), 2)
#         })
#
#     if request.method == "POST":
#         try:
#             for label in antecedents.keys():
#                 val = float(request.form[label])
#                 sim.input[label] = val
#
#             sim.compute()
#             result = sim.output[session['output_column']]
#
#             # Generate graph as base64
#             fig, ax = plt.subplots()
#             consequent.view(sim=sim, ax=ax)
#             buf = io.BytesIO()
#             plt.savefig(buf, format="png")
#             buf.seek(0)
#             graph = base64.b64encode(buf.getvalue()).decode("utf-8")
#             buf.close()
#             plt.close(fig)
#
#         except Exception as e:
#             return f"Error during fuzzy computation: {str(e)}"
#
#     return render_template("fuzzy_graph.html",
#                            antecedents=antecedents.keys(),
#                            metrics=metrics,
#                            graph=graph)

@app.route("/home/mining_technique_option/unsupervised_graph/<method>",methods=["GET","POST"])
def unsupervised_graph(method):
    file = get_current_file()
    numeric_columns = [col for col in file.columns if file[col].dtype in ['int64', 'float64']]
    graph = ""
    palette = ""
    ss = {
        "columns":numeric_columns,
        "graph":graph,
        "method":method,
    }
    methods = ['Kmeans','KMedoids','Hierarchical']
    if method == 'Kmeans':
        palette = 'Set2'
    elif method == 'KMedoids':
        palette = 'coolwarm'
    elif method == 'Hierarchical':
        palette = 'deep'

    if method in methods:
        if request.method == "POST":
            x = request.form.get("x_axis")
            y = request.form.get("y_axis")
            if x and y:
                plt.figure(figsize=(13, 6))
                sns.scatterplot(x=x, y=y, hue=f'Cluster ({method})', data=file, palette=palette)
                plt.title(f"{x} vs {y}({method})")
                plt.grid(True)
                graph = plot_to_base64()
                ss["graph"] = graph
                with open("static/assets/current_displayed_graph.txt", "w") as f:
                    f.write(graph)
            return render_template("unsupervised_graphs.html", **ss)
        return render_template("unsupervised_graphs.html",**ss)
    else:
        return jsonify({"error": {"message": "route doesn't exist"}}), 500

@app.route("/home/mining_technique_option/supervised_graph/<method>/<predicted_column>",methods=["GET","POST"])
def supervised_graph(method,predicted_column):
    file = get_current_file()
    predicted_column = predicted_column
    print(predicted_column)
    numeric_columns = [col for col in file.columns if file[col].dtype in ['int64', 'float64']]
    graph = ""
    ss = {
        "columns":numeric_columns,
        "graph":graph,
        "method":method,
        "predicted_column":predicted_column
    }
    if method == 'knn':
        if request.method == "POST":
            x = request.form.get("x_axis")
            y = request.form.get("y_axis")
            if x and y:
                plt.figure(figsize=(13, 6))
                sns.scatterplot(x=x, y=y, hue=predicted_column, data=file, palette='Set2')
                plt.title(f"{x} vs {y} - {predicted_column}")
                plt.grid(True)
                graph = plot_to_base64()
                ss["graph"] = graph
                with open("static/assets/current_displayed_graph.txt", "w") as f:
                    f.write(graph)
        return render_template("supervised_graphs.html", **ss)
    elif method == 'linear_regression':
        prediction = predicted_column.replace("Predicted_", "").replace("(linear_regression)", "")
        plt.figure(figsize=(13, 6))
        sns.regplot(x=prediction, y=predicted_column,data=file, color='purple', scatter_kws={'s': 50})
        plt.plot([file[prediction].min(), file[prediction].max()], [file[prediction].min(),file[prediction].max()],color='red',linestyle='--', label='Ideal Fit')
        plt.xlabel(f"Actual {prediction}")
        plt.ylabel(f"{predicted_column}")
        plt.title(f"Actual vs. Predicted {prediction} (Linear Regression)")
        plt.grid(True)
        plt.legend()
        graph = plot_to_base64()
        ss["graph"] = graph
        with open("static/assets/current_displayed_graph.txt", "w") as f:
            f.write(graph)
        print(graph)
        return render_template("supervised_graphs.html", **ss)
    elif method == 'decision_tree':
        ss["graph"] = get_current_displayed_graph()
        print(ss)
        return render_template("supervised_graphs.html", **ss)
    else:
        return jsonify({"error": {"message": "route doesn't exist"}}), 500



@app.route("/home/mining_technique_option/evaluation/",methods=["GET","POST"])
def evaluation():
    file = get_current_report()
    ss = {
        "Silhouette_Score":round(file["Value"][0],4),
        "Davies_Bouldin_Index":round(file["Value"][1],4),
    }
    return render_template("eval_methods.html", **ss)

@app.route("/save_graphs", methods=["POST","GET"])
def save_graphs():
    all_graphs_path = get_all_graphs_file_path()
    graph_b64 = get_current_displayed_graph()
    if not graph_b64:
        return jsonify(False)

    if os.path.exists(all_graphs_path):
        all_graphs = pd.read_csv(all_graphs_path)
    else:
        all_graphs = pd.DataFrame(columns=["graphs"])
    existing_graphs = all_graphs["graphs"].tolist()
    print(len(existing_graphs), "-----")
    if request.method == "GET":
        return jsonify(graph_b64 not in existing_graphs)

    if request.method == "POST":
        if graph_b64 not in existing_graphs:
            df = pd.concat([all_graphs, pd.DataFrame({"graphs": [graph_b64]})], ignore_index=True)
            df.to_csv(all_graphs_path, index=False)
            print(len(existing_graphs), end="")
            return jsonify(True)
        else:
            print(len(existing_graphs), end="")
            return jsonify(False)


@app.route("/home/mining_technique_option/visualization_method",methods=["GET","POST"])
def visualization_method():
    with open("static/assets/current_displayed_graph.txt", "w") as file:
        file.write("")
    method = request.form.get("chart_type")
    if request.method == "POST":
        if method == "bar":
            return redirect(url_for("xy_graphs",chart = method))
        elif method == "stacked_bar":
            return redirect(url_for("xy_graphs",chart = method))
        elif method == "boxplot":
            return redirect(url_for("xy_graphs",chart = method))
        elif method == "multi_set_bar":
            return redirect(url_for("xy_graphs",chart = method))
        elif method == "scatter":
            return redirect(url_for("xy_graphs",chart = method))
        elif method == "histogram":
            return redirect(url_for("xy_graphs",chart = method))
        elif method == "pie":
            return redirect(url_for("pie_graph"))
        else:
            return jsonify({"error": {"message": "route doesn't exist"}}), 500
    return render_template("visualization_method.html")


@app.route("/home/mining_technique_option/visualization_method/<chart>", methods=["GET", "POST"])
def xy_graphs(chart):
    file = get_current_file()

    if file.empty or not isinstance(file, pd.DataFrame):
        return jsonify({"error": {"message": "No valid data available to plot"}}), 400

    numeric_columns = [col for col in file.columns if file[col].dtype in ['int64', 'float64']]

    if chart == "bar":
        ss = {
            "method": chart,
            "x_columns": file.columns,
            "y_columns": numeric_columns,
            "color_columns": file.columns,
            "hue_columns": file.columns,
            "graph":"",
        }
        if request.method == "POST":
            x_axis = request.form.get("x-column")
            y_axis = request.form.get("y-column")
            x_axis_label = request.form.get("x-label")
            y_axis_label = request.form.get("y-label")
            y_agg_func = request.form.get('y-agg', 'none')
            color_column = request.form.get('color-column')
            color_agg_func = request.form.get('color-agg', 'none')
            hue_column = request.form.get('hue-column')

            if not x_axis or x_axis not in file.columns:
                return jsonify({"error": {"message": "Invalid or missing X-axis column"}}), 400
            if not y_axis or y_axis not in file.columns:
                return jsonify({"error": {"message": "Invalid or missing Y-axis column"}}), 400
            if color_column and color_column not in file.columns:
                return jsonify({"error": {"message": "Invalid color column"}}), 400
            if hue_column and hue_column not in file.columns:
                return jsonify({"error": {"message": "Invalid hue column"}}), 400

            def apply_agg(group, func):
                if func == 'count':
                    return len(group)
                elif func == 'count_unique':
                    return group.nunique()
                elif func == 'average':
                    return group.mean()
                elif func == 'max':
                    return group.max()
                elif func == 'min':
                    return group.min()
                elif func == 'sum':
                    return group.sum()
                elif func == 'median':
                    return group.median()
                return group

            if y_agg_func != 'none':
                y_values = file.groupby(x_axis)[y_axis].agg(lambda x: apply_agg(x, y_agg_func))
            else:
                y_values = file.groupby(x_axis)[y_axis].first()

            if color_column and color_agg_func != 'none':
                color_values = file.groupby(x_axis)[color_column].agg(lambda x: apply_agg(x, color_agg_func))
            else:
                color_values = None

            fig, ax = plt.subplots(figsize=(12, 6))
            if hue_column:
                sns.barplot(x=x_axis, y=y_axis, hue=hue_column, data=file, ax=ax,estimator=lambda x: apply_agg(x, y_agg_func))
            else:
                y_values.plot(kind='bar', ax=ax)
                if color_values is not None:
                    norm_colors = color_values / (color_values.max() if color_values.max() != 0 else 1)
                    colors = plt.cm.viridis(norm_colors)
                    for i, bar in enumerate(ax.patches):
                        bar.set_color(colors[i])
            print("y",y_axis_label)
            print("x",x_axis_label)
            if x_axis_label != "":
                ax.set_xlabel(x_axis_label)
            else:
                ax.set_xlabel(x_axis)

            if y_axis_label != "":
                ax.set_ylabel(y_axis_label)
            else:
                ax.set_ylabel(f"{y_agg_func.capitalize() if y_agg_func != 'none' else ''} {y_axis}")

            ax.set_title(f"{y_agg_func.capitalize() if y_agg_func != 'none' else 'Raw'} {y_axis} by {x_axis}"
                         f"{' (Hue by ' + hue_column if hue_column else ''}"
                         f"{' (Color by ' + color_agg_func.capitalize() + ' of ' + color_column if color_values is not None else ''}")
            graph = plot_to_base64()
            with open("static/assets/current_displayed_graph.txt", "w") as f:
                f.write(graph)
            ss["graph"] = graph
        return render_template("vis.html", **ss)
    elif chart == "stacked_bar":
        ss = {
            "method": chart,
            "x_columns": file.columns,
            "y_columns": numeric_columns,
            "hue_columns": file.columns,
            "graph":"",
        }
        if request.method == "POST":
            x_axis = request.form.get("x-column")
            x_axis_label = request.form.get("x-label")
            y_axis_label = request.form.get("y-label")
            hue_column = request.form.get("hue-column")
            y_selections = request.form.get("y-selections")
            print(f"y_selections from form: {y_selections}")

            if not x_axis or x_axis not in file.columns:
                return jsonify({"error": {"message": "Invalid or missing X-axis column"}}), 400
            if not y_selections:
                return jsonify({"error": {"message": "Invalid or missing Y-axis selections"}}), 400

            try:
                y_selections = json.loads(y_selections)
            except json.JSONDecodeError:
                return jsonify({"error": {"message": "Invalid Y-axis selections format"}}), 400

            for sel in y_selections:
                if sel["column"] not in file.columns:
                    return jsonify({"error": {"message": f"Invalid Y-axis column: {sel['column']}"}}), 400

            def apply_agg(group, func):
                if func == "count":
                    return len(group)
                elif func == "count_unique":
                    return group.nunique()
                elif func == "average":
                    return group.mean()
                elif func == "max":
                    return group.max()
                elif func == "min":
                    return group.min()
                elif func == "sum":
                    return group.sum()
                elif func == "median":
                    return group.median()
                return group

            grouped_data = []
            for sel in y_selections:
                y_axis = sel["column"]
                y_agg_func = sel["agg"]
                if hue_column:
                    grouped = file.groupby([x_axis, hue_column])[y_axis].agg(
                        lambda x: apply_agg(x, y_agg_func)).unstack(fill_value=0)
                    grouped.columns = [f"{y_axis} ({y_agg_func}) - {hue_val}" for hue_val in grouped.columns]
                else:
                    grouped = file.groupby(x_axis)[y_axis].agg(lambda x: apply_agg(x, y_agg_func)).to_frame()
                    grouped.columns = [f"{y_axis} ({y_agg_func})"]

                grouped_data.append(grouped)
            if grouped_data:
                final_data = pd.concat(grouped_data, axis=1)
            else:
                return jsonify({"error": {"message": "No data to plot after aggregation"}}), 400

            # Plot the stacked bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            final_data.plot(kind="bar", stacked=True, ax=ax)


            if x_axis_label != "":
                ax.set_xlabel(x_axis_label)
            else:
                ax.set_xlabel(x_axis)

            if y_axis_label != "":
                ax.set_ylabel(y_axis_label)
            else:
                ax.set_ylabel("Values")

            title = f"{y_agg_func.capitalize() if y_agg_func != 'none' else 'Raw'} {y_axis} by {x_axis}"
            if hue_column:
                title += f" (Hue by {hue_column})"
            ax.set_title(title)

            graph = plot_to_base64()
            with open("static/assets/current_displayed_graph.txt", "w") as f:
                f.write(graph)
            ss["graph"] = graph

        return render_template("vis.html", **ss)
    elif chart == "boxplot":
        ss = {
            "method": chart,
            "x_columns": file.columns,
            "y_columns": numeric_columns,
            "color_columns": file.columns,
            "hue_columns": file.columns,
        }
        if request.method == "POST":
            x_axis = request.form.get("x-column")
            y_axis = request.form.get("y-column")
            x_axis_label = request.form.get("x-label")
            y_axis_label = request.form.get("y-label")
            hue_column = request.form.get("hue-column")

            if not y_axis or y_axis not in numeric_columns:
                return jsonify({"error": {"message": "Invalid or missing Y-axis column (must be numeric)"}}), 400
            if x_axis and x_axis not in file.columns:
                return jsonify({"error": {"message": "Invalid X-axis column"}}), 400
            if hue_column and hue_column not in file.columns:
                return jsonify({"error": {"message": "Invalid hue column"}}), 400

            plot_data = file if not x_axis else file[[x_axis, y_axis, hue_column] if hue_column else [x_axis, y_axis]]

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(data=plot_data, x=x_axis if x_axis else None, y=y_axis, hue=hue_column if hue_column else None,ax=ax)

            if x_axis_label != "":
                ax.set_xlabel(x_axis_label)
            elif x_axis:
                ax.set_xlabel(x_axis)
            else:
                ax.set_xlabel("")

            if y_axis_label != "":
                ax.set_ylabel(y_axis_label)
            else:
                ax.set_ylabel(y_axis)

            ax.set_title(f"Boxplot of {y_axis}" + (f" by {x_axis}" if x_axis else "") + (
                f" (Hue by {hue_column})" if hue_column else ""))

            plt.tight_layout()

            graph = plot_to_base64()
            with open("static/assets/current_displayed_graph.txt", "w") as f:
                f.write(graph)
            ss["graph"] = graph
        return render_template("vis.html", **ss)
    elif chart == "multi_set_bar":
        ss = {
            "method": chart,
            "x_columns": file.columns,
            "y_columns": numeric_columns,
            "hue_columns": file.columns,
            "graph":""
        }
        if request.method == "POST":
            x_axis = request.form.get("x-column")
            x_axis_label = request.form.get("x-label")
            y_axis_label = request.form.get("y-label")
            hue_column = request.form.get("hue-column")
            y_selections = request.form.get("y-selections")
            print(f"y_selections from form: {y_selections}")  # Debug print

            if not x_axis or x_axis not in file.columns:
                return jsonify({"error": {"message": "Invalid or missing X-axis column"}}), 400
            if not y_selections:
                return jsonify({"error": {"message": "Invalid or missing Y-axis selections"}}), 400

            try:
                y_selections = json.loads(y_selections)
                print(f"Parsed y_selections: {y_selections}")  # Debug print
            except json.JSONDecodeError:
                return jsonify({"error": {"message": "Invalid Y-axis selections format"}}), 400

            if not y_selections:
                return jsonify({"error": {"message": "At least one Y-axis column must be selected"}}), 400

            for sel in y_selections:
                if sel["column"] not in file.columns:
                    return jsonify({"error": {"message": f"Invalid Y-axis column: {sel['column']}"}}), 400

            def apply_agg(group, func):
                if func == "count":
                    return len(group)
                elif func == "count_unique":
                    return group.nunique()
                elif func == "average":
                    return group.mean()
                elif func == "max":
                    return group.max()
                elif func == "min":
                    return group.min()
                elif func == "sum":
                    return group.sum()
                elif func == "median":
                    return group.median()
                return group

            grouped_data = []
            for sel in y_selections:
                y_axis = sel["column"]
                y_agg_func = sel["agg"]
                print(f"Processing y_axis: {y_axis}, agg: {y_agg_func}")  # Debug print
                if hue_column and hue_column in file.columns:
                    grouped = file.groupby([x_axis, hue_column])[y_axis].agg(lambda x: apply_agg(x, y_agg_func)).unstack(fill_value=0)
                    grouped.columns = [f"{y_axis} ({y_agg_func}) - {hue_val}" for hue_val in grouped.columns]
                else:
                    grouped = file.groupby(x_axis)[y_axis].agg(lambda x: apply_agg(x, y_agg_func))
                    grouped.name = f"{y_axis} ({y_agg_func})"
                grouped_data.append(grouped)
            print(f"Grouped data: {grouped_data}")

            if grouped_data:
                final_data = pd.concat(grouped_data, axis=1)
                print(f"Final data shape: {final_data.shape}")
            else:
                return jsonify({"error": {"message": "No data to plot after aggregation"}}), 400

            fig, ax = plt.subplots(figsize=(12, 6))
            if hue_column and hue_column in file.columns:
                width = 0.35
                x = range(len(final_data.index))
                for i, column in enumerate(final_data.columns):
                    offset = width * i
                    ax.bar([xi + offset for xi in x], final_data[column], width, label=column)
                ax.set_xticks([xi + width * (len(final_data.columns) - 1) / 2 for xi in x])
                ax.set_xticklabels(final_data.index)
            else:
                final_data.plot(kind="bar", ax=ax)

            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

            if x_axis_label != "":
                ax.set_xlabel(x_axis_label)
            else:
                ax.set_xlabel(x_axis)

            if y_axis_label != "":
                ax.set_ylabel(y_axis_label)
            else:
                ax.set_ylabel("Values")

            ax.set_title(f"Multi-Set Bar: Multiple Metrics by {x_axis}" + (f" (Hue by {hue_column})" if hue_column else ""))

            graph = plot_to_base64()
            with open("static/assets/current_displayed_graph.txt", "w") as f:
                f.write(graph)
            ss["graph"] = graph

        return render_template("vis.html", **ss)
    elif chart == "scatter":
        ss = {
            "method": chart,
            "x_columns": numeric_columns,
            "y_columns": numeric_columns,
            "color_columns": file.columns,
            "hue_columns": file.columns,
        }
        if request.method == "POST":
            x_axis = request.form.get("x-column")
            y_axis = request.form.get("y-column")
            x_axis_label = request.form.get("x-label")
            y_axis_label = request.form.get("y-label")
            hue_column = request.form.get("hue-column")
            color_column = request.form.get("color-column")

            if not x_axis or x_axis not in numeric_columns:
                return jsonify({"error": {"message": "Invalid or missing X-axis column (must be numeric)"}}), 400
            if not y_axis or y_axis not in numeric_columns:
                return jsonify({"error": {"message": "Invalid or missing Y-axis column (must be numeric)"}}), 400
            if hue_column and hue_column not in file.columns:
                return jsonify({"error": {"message": "Invalid hue column"}}), 400
            if color_column and color_column not in file.columns:
                return jsonify({"error": {"message": "Invalid color column"}}), 400

            plot_data = file[[x_axis, y_axis]]
            if hue_column:
                plot_data[hue_column] = file[hue_column]
            if color_column:
                plot_data[color_column] = file[color_column]

            fig, ax = plt.subplots(figsize=(12, 6))
            if hue_column and color_column:
                sns.scatterplot(data=plot_data, x=x_axis, y=y_axis, hue=hue_column, size=color_column, ax=ax)
            elif hue_column:
                sns.scatterplot(data=plot_data, x=x_axis, y=y_axis, hue=hue_column, ax=ax)
            elif color_column:
                sns.scatterplot(data=plot_data, x=x_axis, y=y_axis, size=color_column, ax=ax)
            else:
                sns.scatterplot(data=plot_data, x=x_axis, y=y_axis, ax=ax)

            if x_axis_label != "":
                ax.set_xlabel(x_axis_label)
            else:
                ax.set_xlabel(x_axis)

            if y_axis_label != "":
                ax.set_ylabel(y_axis_label)
            else:
                ax.set_ylabel(y_axis)

            ax.set_title(f"Scatter Plot: {y_axis} vs {x_axis}" + (f" (Hue by {hue_column})" if hue_column else "") + (
                f" (Size by {color_column})" if color_column else ""))

            plt.tight_layout()

            graph = plot_to_base64()
            with open("static/assets/current_displayed_graph.txt", "w") as f:
                f.write(graph)
            ss["graph"] = graph
        return render_template("vis.html", **ss)
    elif chart == "histogram":
        ss = {
            "method": chart,
            "x_columns": file.columns,
            "hue_columns": file.columns,
        }
        if request.method == "POST":
            x_axis = request.form.get("x-column")
            x_axis_label = request.form.get("x-label")
            hue_column = request.form.get("hue-column")

            if not x_axis or x_axis not in file.columns:
                return jsonify({"error": {"message": "Invalid or missing X-axis column"}}), 400
            if hue_column and hue_column not in file.columns:
                return jsonify({"error": {"message": "Invalid hue column"}}), 400

            plot_data = file[[x_axis]]
            if hue_column:
                plot_data[hue_column] = file[hue_column]

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(data=plot_data, x=x_axis, hue=hue_column if hue_column else None, ax=ax, kde=True)

            if x_axis_label != "":
                ax.set_xlabel(x_axis_label)
            else:
                ax.set_xlabel(x_axis)

            ax.set_ylabel("Count")
            ax.set_title(f"Histogram of {x_axis}" + (f" (Hue by {hue_column})" if hue_column else ""))

            plt.tight_layout()

            graph = plot_to_base64()
            with open("static/assets/current_displayed_graph.txt", "w") as f:
                f.write(graph)
            ss["graph"] = graph
        return render_template("vis.html", **ss)
    else:
        return jsonify({"error": {"message": "route doesn't exist"}}), 500

@app.route("/home/mining_technique_option/visualization_method/pie", methods=["GET", "POST"])
def pie_graph():
    file = get_current_file()

    if file.empty or not isinstance(file, pd.DataFrame):
        return jsonify({"error": {"message": "No valid data available to plot"}}), 400

    numeric_columns = [col for col in file.columns if file[col].dtype in ['int64', 'float64']]
    column_types = {col: 'numeric' if col in numeric_columns else 'non-numeric' for col in file.columns}

    ss = {
        "arc_columns": file.columns,
        "hue_columns": file.columns,
        "column_types": column_types,
        "graph": "",
    }

    if request.method == "POST":
        arc_column = request.form.get("arc-column")
        arc_agg = request.form.get("arc-agg", "count")
        hue_column = request.form.get("hue-column")

        if not arc_column or arc_column not in file.columns:
            return jsonify({"error": {"message": "Invalid or missing arc column"}}), 400
        if hue_column and hue_column not in file.columns:
            return jsonify({"error": {"message": "Invalid hue column"}}), 400

        if arc_column not in numeric_columns and arc_agg != "count":
            return jsonify({"error": {"message": "Non-numeric columns can only use 'count' aggregation"}}), 400

        def apply_agg(group, func):
            if func == "none":
                return group.iloc[0] if not group.empty else 0
            elif func == "count":
                return len(group)
            elif func == "count_unique":
                return group.nunique()
            elif func == "average":
                return group.mean()
            elif func == "max":
                return group.max()
            elif func == "min":
                return group.min()
            elif func == "sum":
                return group.sum()
            elif func == "median":
                return group.median()
            return group

        if hue_column:
            if arc_agg == "count":
                grouped = file.groupby([arc_column, hue_column]).size().unstack(fill_value=0)
            elif arc_agg == "none":
                # For "none", take the first value in each group (if needed for hue)
                grouped = file.groupby([arc_column, hue_column])[arc_column].first().unstack(fill_value=0)
            else:
                grouped = file.groupby([arc_column, hue_column])[arc_column].agg(
                    lambda x: apply_agg(x, arc_agg)).unstack(fill_value=0)
        else:
            if arc_agg == "count":
                grouped = file.groupby(arc_column).size()
            elif arc_agg == "none":
                grouped = file.groupby(arc_column)[arc_column].first()
            else:
                grouped = file.groupby(arc_column)[arc_column].agg(lambda x: apply_agg(x, arc_agg))

        fig, ax = plt.subplots(figsize=(6, 6))
        if hue_column:
            outer = grouped.sum(axis=1)
            inner = grouped
            ax.pie(outer, labels=outer.index, autopct='%1.1f%%', startangle=90, radius=1.0)
            ax.pie(inner.values.flatten(),
                   labels=[f"{outer_label}-{inner_label}" for outer_label, inner_label in inner.stack().index],
                   autopct='%1.1f%%', startangle=90, radius=0.7)
        else:
            ax.pie(grouped, labels=grouped.index, autopct='%1.1f%%', startangle=90)

        ax.set_title(f"Pie Chart of {arc_column} ({arc_agg})" + (f" with Hue by {hue_column}" if hue_column else ""))
        ax.axis('equal')

        graph = plot_to_base64()
        with open("static/assets/current_displayed_graph.txt", "w") as f:
            f.write(graph)
        ss["graph"] = graph

    return render_template("pie_chart.html", **ss)


@app.route("/home/mining_technique_option/visualization_method/download_pdf",methods=["GET","POST"])
def download_pdf():
    return render_template("download_pdf.html")


@app.route("/generate_pdf", methods=["GET"])
def generate_pdf():
    PDF_PATH = f"static/assets/outputs/f{get_current_dataset_name()}.pdf"
    CSV_FILE = get_all_graphs_file_path()
    IMAGE_COLUMN = 'graphs'
    try:
        os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)
        df = pd.read_csv(CSV_FILE)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        for idx, row in df.iterrows():
            base64_data = row[IMAGE_COLUMN]
            if "," in base64_data:
                base64_data = base64_data.split(",")[1]
            try:
                img_data = base64.b64decode(base64_data)
                image = Image.open(BytesIO(img_data))
                temp_path = f"temp_image_{idx}.png"
                image.save(temp_path)
                pdf.add_page()
                pdf.image(temp_path, x=10, y=10, w=pdf.w - 20)
                os.remove(temp_path)
            except Exception as e:
                print(f"Skipping image {idx} due to error: {e}")

        pdf.output(PDF_PATH)
        return jsonify({"status": "ready"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/download_ready_pdf")
def download_ready_pdf():
    PDF_PATH = f"static/assets/outputs/f{get_current_dataset_name()}.pdf"
    if os.path.exists(PDF_PATH):
        return send_file(PDF_PATH, as_attachment=True, download_name=f"{get_current_dataset_name()}.pdf")
    return "PDF not found", 404

@app.route("/download_dataset")
def download_dataset():
    main_file_path = get_current_path()
    test_file_path = get_test_data_file()

    main_file_exists = os.path.exists(main_file_path)
    test_file_exists = os.path.exists(test_file_path) and os.path.getsize(test_file_path) > 0

    return render_template("download_file.html",
                           main_file_exists=main_file_exists,
                           test_file_exists=test_file_exists)

@app.route("/download/<file_type>")
def download_file(file_type):
    ext = get_current_file_extension()
    if file_type == "main":
        file_path = get_current_path()
        download_name = f"{get_current_dataset_name()}.{ext}"
    elif file_type == "test":
        file_path = get_test_data_file()
        download_name = f"test_data.{ext}"
    else:
        return "Invalid file type", 400

    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=download_name)
    else:
        return "File not found", 404

if __name__ == '__main__':
     app.run(debug=True ,port=5005)