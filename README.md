# Fetal Brain Anomalies Detection

## Introduction

This project aims to detect fetal brain anomalies using the YOLOv5 model and provides a web interface for easy image analysis. It is designed to assist medical professionals and researchers in identifying anomalies in fetal brain images, enabling timely diagnosis and intervention.

## To Run the code Locally:

1. Clone the Repository to your local machine:

```shell
git clone https://github.com/Vedansh-777/Fetal_Brain_Anomalies.git
```
2. Create a virtual environment and activate it:

```python
python -m venv myenv 
```
Assuming you're on  macOS or Linux (activate):
```python
source myenv/bin/activate 
```

3. Install Dependencies:

Install all the dependencies listed in _requirements.txt_ file.
```python
pip install -r requirements.txt
```

4. Change Directory (optional):

Navigate to the directory where _manage.py_ is present:
> Replace 'replace_directory' with the folder where `manage.py` is present (maybe _Django_Backend_)
```python
cd replace_directory
```

5. Apply Database Migrations:

Apply database migrations to set up the database:
```python
python manage.py migrate
```

6. Create Super User:

Create a superuser account to access the admin panel and manage the application:
```python
python manage.py createsuperuser
```
Follow the prompts to set up the superuser credentials.
> Note: You need to have an image named _**default.png**_ in the '_media_' folder for user profiles. If not present, create a '_media_' folder and add an image file named '_default.png_' if it does not exist.


7. Run the Application Locally:

Start the local development server:
```python
python manage.py runserver
```
You can access the web interface by visiting http://127.0.0.1:8000 in your web browser and logging in with the superuser credentials.


### Usage

1. Create an ImageSet: Before uploading images for analysis, create an ImageSet from the ImageSet detail page.

2. Upload Images: After creating an ImageSet, you can upload images into the ImageSet from the ImageSet detail page.

3. Detect Objects: On the images list page, click on the "Detect Object" button.

4. Select YOLOv5 Model: Choose a YOLOv5 model for object detection. Dependencies and pre-trained models will be downloaded automatically.


## Identified Classes

The following table shows the classes identified in the project, along with relevant statistics:

| Class                      | Images | Instances | Box(P) | Recall | mAP50 | mAP50-95 | Mask(P) | Recall | mAP50 | mAP50-95 |
| -------------------------- | ------ | --------- | ------ | ------ | ----- | -------- | ------- | ------ | ----- | -------- |
| All                        | 357    | 381       | 0.901  | 0.942  | 0.962 | 0.637    | 0.896   | 0.935  | 0.957 | 0.55     |
| Arnold Chiari Malformation | 357    | 10        | 0.935  | 1      | 0.995 | 0.679    | 0.935   | 1      | 0.995 | 0.505    |
| Arachnoid Cyst             | 357    | 22        | 1      | 0.968  | 0.995 | 0.662    | 1       | 0.968  | 0.995 | 0.62     |
| Cerebellar Hypoplasia      | 357    | 32        | 0.844  | 0.845  | 0.897 | 0.633    | 0.875   | 0.876  | 0.943 | 0.583    |
| Cisterna Magna             | 357    | 10        | 0.789  | 1      | 0.977 | 0.599    | 0.71    | 0.9    | 0.887 | 0.479    |
| Colphocephaly              | 357    | 29        | 0.802  | 0.724  | 0.842 | 0.438    | 0.802   | 0.724  | 0.842 | 0.429    |
| Encephalocele              | 357    | 37        | 1      | 0.915  | 0.953 | 0.652    | 1       | 0.915  | 0.953 | 0.631    |
| Holoprosencephaly          | 357    | 4         | 0.862  | 1      | 0.995 | 0.846    | 0.862   | 1      | 0.995 | 0.647    |
| Hydracenphaly              | 357    | 8         | 0.91   | 1      | 0.995 | 0.765    | 0.91    | 1      | 0.995 | 0.728    |
| Intracranial Hemorrhage    | 357    | 18        | 1      | 0.94   | 0.972 | 0.505    | 1       | 0.94   | 0.972 | 0.545    |
| Intracranial Tumor         | 357    | 1         | 0.781  | 1      | 0.995 | 0.697    | 0.781   | 1      | 0.995 | 0.298    |
| Mild Ventriculomegaly      | 357    | 72        | 0.956  | 0.907  | 0.935 | 0.56     | 0.956   | 0.907  | 0.942 | 0.502    |
| Moderate Ventriculomegaly  | 357    | 71        | 0.79   | 0.956  | 0.951 | 0.646    | 0.79    | 0.956  | 0.951 | 0.599    |
| Polencephaly               | 357    | 32        | 1      | 0.968  | 0.988 | 0.588    | 1       | 0.968  | 0.988 | 0.549    |

<br>

Feel free to reach out if you have any questions or need further assistance. Enjoy using the Fetal Brain Anomalies Detection application!

---
