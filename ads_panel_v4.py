import cv2
from keras import models
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QDateTime
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
import sqlite3

# Load gender model
gender_model = models.load_model('modells/network_gender13.keras')

# Load age model
age_model = models.load_model('modells/network_ages14_v2.keras')

# Load emotion model
emotion_model = models.load_model('modells/network_emotion8.keras')


# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


class Persons:
    changed = True
    persons = {
        "female": False,
        "male": False,
        "child": False,
        "young_adult": False,
        "middle_aged": False,
        "elderly": False,
        "happy": False,
        "neutral": False,
        "sad": False
    }

    female_ls = []
    male_ls = []
    child_ls = []
    young_adult_ls = []
    middle_aged_ls = []
    elderly_ls = []
    happy_ls = []
    neutral_ls = []
    sad_ls = []



class UpdateVideo:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)  # Set width
        self.cap.set(4, 480)  # Set height

    def predict_gender(self, img):
        input_image = np.expand_dims(img / 255.0, axis=0)
        pred = gender_model.predict(input_image)
        test_pred_bool = np.rint(pred)
        if test_pred_bool[0][0] == 1:
            return "Woman"
        else:
            return "Man"

    def predict_age(self, img):
        input_image = np.expand_dims(img / 255.0, axis=0)
        pred = age_model.predict(input_image)
        test_pred_bool = np.argmax(pred, axis=1)
        if test_pred_bool[0] == 0:
            return "Child"
        elif test_pred_bool[0] == 1:
            return "Young adult"
        elif test_pred_bool[0] == 2:
            return "Middle aged"
        else:
            return "Elderly"

    def predict_emotion(self, img):
        input_image = np.expand_dims(img / 255.0, axis=0)
        pred = emotion_model.predict(input_image)
        test_pred_bool = np.argmax(pred, axis=1)
        if test_pred_bool[0] == 0:
            return "Happy"
        elif test_pred_bool[0] == 1:
            return "Neutral"
        else:
            return "Sad"

    def details_get_frame(self):
        _, video_img = self.cap.read()
        gray = cv2.cvtColor(video_img, cv2.COLOR_BGR2GRAY)
        video_img = cv2.cvtColor(video_img, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        previous_persons = Persons.persons.copy()

        for key in Persons.persons:
            Persons.persons[key] = False

        """Persons.female = 0
        Persons.male = 0
        Persons.child = 0
        Persons.young_adult = 0
        Persons.middle_aged = 0
        Persons.elderly = 0"""

        female = False
        male = False
        child = False
        young_adult = False
        middle_aged = False
        elderly = False
        sad = False
        neutral = False
        happy = False

        for (x, y, w, h) in faces:
            cropped_img = video_img[y:y + h, x:x + w]
            resized_cropped_img = cv2.resize(cropped_img, (200, 200))
            gender = self.predict_gender(resized_cropped_img)
            age = self.predict_age(resized_cropped_img)
            resized_cropped_img = cv2.resize(cropped_img, (48, 48))
            resized_cropped_img_gray = cv2.cvtColor(resized_cropped_img, cv2.COLOR_RGB2GRAY)
            emotion = self.predict_emotion(resized_cropped_img_gray)

            if gender == "Woman":
                female = True
            elif gender == "Man":
                male = True

            if age == "Child":
                child = True
            elif age == "Young adult":
                young_adult = True
            elif age == "Middle aged":
                middle_aged = True
            elif age == "Elderly":
                elderly = True

            if emotion == "Happy":
                happy = True
            elif emotion == "Neutral":
                neutral = True
            elif emotion == "Sad":
                sad = True

            cv2.rectangle(video_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(video_img, gender + " " + age + " " + emotion, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        height, width, channel = video_img.shape
        bytes_per_line = 3 * width
        q_video_img = QImage(video_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap_video = QPixmap.fromImage(q_video_img)

        #update the list

        Persons.female_ls.append(female)
        Persons.male_ls.append(male)
        Persons.child_ls.append(child)
        Persons.young_adult_ls.append(young_adult)
        Persons.middle_aged_ls.append(middle_aged)
        Persons.elderly_ls.append(elderly)
        Persons.sad_ls.append(sad)
        Persons.neutral_ls.append(neutral)
        Persons.happy_ls.append(happy)

        if len(Persons.female_ls) >= 50:
            Persons.female_ls.pop(0)
            Persons.male_ls.pop(0)
            Persons.child_ls.pop(0)
            Persons.young_adult_ls.pop(0)
            Persons.middle_aged_ls.pop(0)
            Persons.elderly_ls.pop(0)
            Persons.sad_ls.pop(0)
            Persons.neutral_ls.pop(0)
            Persons.happy_ls.pop(0)

        #gender
        Persons.persons["female"] = max(Persons.female_ls, key=Persons.female_ls.count)
        Persons.persons["male"] = max(Persons.male_ls, key=Persons.male_ls.count)

        #age
        Persons.persons["child"] = max(Persons.child_ls, key=Persons.child_ls.count)
        Persons.persons["young_adult"] = max(Persons.young_adult_ls, key=Persons.young_adult_ls.count)
        Persons.persons["middle_aged"] = max(Persons.middle_aged_ls, key=Persons.middle_aged_ls.count)
        Persons.persons["elderly"] = max(Persons.elderly_ls, key=Persons.elderly_ls.count)

        #emotion
        Persons.persons["sad"] = max(Persons.sad_ls, key=Persons.sad_ls.count)
        Persons.persons["happy"] = max(Persons.happy_ls, key=Persons.happy_ls.count)
        Persons.persons["neutral"] = max(Persons.neutral_ls, key=Persons.neutral_ls.count)


        if Persons.persons == previous_persons:
            Persons.changed = False
        else:
            Persons.changed = True

        # print("male: "+str(Persons.persons["male"]))
        # print("changed: "+str(Persons.changed))

        return pixmap_video

    def ads_get_frame(self):
        conn = sqlite3.connect('picturesdatabase.db')
        cursor = conn.cursor()

        after_where = ""

        for key, value in Persons.persons.items():
            if value:
                after_where += (" " + str(key) + " = 1 AND ")

        if after_where == "":
            file_url = "pictures/default.png"
        else:
            select = "SELECT filename FROM images WHERE" + after_where[:-4] + "ORDER BY random() LIMIT 1"
            file_url_record = cursor.execute(select).fetchone()
            if file_url_record:
                file_url = file_url_record[0]
            else:
                file_url = "pictures/default.png"

        # Load advertisement image
        ads_img_bgr = cv2.imread(file_url, cv2.IMREAD_COLOR)
        ads_img = cv2.cvtColor(ads_img_bgr, cv2.COLOR_BGR2RGB)
        height, width, channel = ads_img.shape

        # Reshape image
        if width > 1280:
            new_width = 1280
            new_height = int((new_width / width) * height)

            if new_height > 720:
                new_new_height = 720
                new_new_width = int((new_new_height / new_height) * new_width)
                new_height = new_new_height
                new_width = new_new_width

            ads_img = cv2.resize(ads_img, (new_width, new_height))
            height = new_height
            width = new_width

        bytes_per_line = 3 * width
        q_ads_img = QImage(ads_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap_img = QPixmap.fromImage(q_ads_img)

        return pixmap_img

    def release(self):
        self.cap.release()


class DetailsWindow(QMainWindow):
    def __init__(self, update_video):
        super().__init__()

        self.setWindowTitle("Properties window")
        self.setGeometry(100, 100, 640, 480)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignHCenter)
        self.central_widget.setLayout(self.layout)

        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        self.update_video = update_video

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_feed)
        self.timer.start(30)

    def update_video_feed(self):
        video_frame = self.update_video.details_get_frame()
        self.video_label.setPixmap(video_frame)

    def closeEvent(self, event):
        self.update_video.release()
        super().closeEvent(event)


class AdvertisementWindow(QMainWindow):

    def __init__(self, update_video_feed):
        super().__init__()

        self.setWindowTitle("Advertisement Window")
        self.setGeometry(0, 0, 1920, 1020)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignHCenter)
        self.central_widget.setLayout(self.layout)

        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        self.camera_manager = update_video_feed

        self.last_change_time = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(30)

        self.previous_persons = {
            "female": False,
            "male": False,
            "child": False,
            "young_adult": False,
            "middle_aged": False,
            "elderly": False,
            "happy": False,
            "neutral": False,
            "sad": False
        }

    def update_image(self):

        if self.timer.interval() == 30:
            self.timer.setInterval(3000)
            self.last_change_time = QDateTime.currentDateTime()
            img_frame = self.camera_manager.ads_get_frame()
            self.image_label.setPixmap(img_frame)
            return

        if Persons.persons == self.previous_persons:
            changed = False
        else:
            changed = True

        if changed:
            self.last_change_time = QDateTime.currentDateTime()
            img_frame = self.camera_manager.ads_get_frame()
            self.image_label.setPixmap(img_frame)
        elif self.last_change_time.secsTo(QDateTime.currentDateTime()) >= 30:
            self.last_change_time = QDateTime.currentDateTime()
            img_frame = self.camera_manager.ads_get_frame()
            self.image_label.setPixmap(img_frame)



        self.previous_persons = Persons.persons.copy()

        # print(self.last_change_time.secsTo(QDateTime.currentDateTime()))
        # print("male: "+str(Persons.persons["male"]))
        # print("female: " + str(Persons.persons["female"]))
        # print("changed: "+str(Persons.changed))



if __name__ == '__main__':
    app = QApplication([])

    update_video = UpdateVideo()
    details_window = DetailsWindow(update_video)
    advertisement_window = AdvertisementWindow(update_video)

    details_window.show()
    advertisement_window.show()

    app.exec_()
