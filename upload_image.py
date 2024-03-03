# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'upload_image.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QAbstractItemView
from PyQt5.QtGui import QImage, QPixmap
import cv2
import sqlite3
from datetime import datetime
import os


class Ui_Upload_Image_Window(object):

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setEnabled(True)
        Dialog.resize(1280, 720)
        Dialog.setStyleSheet("QDialog {background-color:rgb(248,245,244);} ")
        self.centralwidget = QtWidgets.QWidget(Dialog)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setStyleSheet("QPushButton {color: #a86777; font-weight:600; background-color: #ffffff; border: 3px solid #a86777; border-radius:20px;} QPushButton:hover{color: #ffffff; background-color: #a86777;} QListWidget::item:selected{color: #ffffff; background-color:#a86777;} QListWidget::item:hover{color:#000000; background-color:#c2949f;}")

        # ad name block
        self.ad_name_label = QtWidgets.QLabel(self.centralwidget)
        self.ad_name_label.setGeometry(QtCore.QRect(80, 30, 381, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.ad_name_label.setFont(font)
        self.ad_name_label.setObjectName("ad_name_label")

        # ad name line edit block
        self.ad_name_lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.ad_name_lineEdit.setEnabled(True)
        self.ad_name_lineEdit.setGeometry(QtCore.QRect(80, 50, 421, 40))
        self.ad_name_lineEdit.setObjectName("ad_name_lineEdit")
        self.ad_name_error_label = QtWidgets.QLabel(self.centralwidget)
        self.ad_name_error_label.setGeometry(QtCore.QRect(80, 92, 321, 16))
        self.ad_name_error_label.setStyleSheet("color: rgb(227, 0, 4)")
        self.ad_name_error_label.setObjectName("ad_name_error_label")

        # image upload block
        self.image_upload_label = QtWidgets.QLabel(self.centralwidget)
        self.image_upload_label.setGeometry(QtCore.QRect(80, 120, 381, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.image_upload_label.setFont(font)
        self.image_upload_label.setObjectName("image_upload_label")

        # browse file
        self.browse_lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.browse_lineEdit.setEnabled(True)
        self.browse_lineEdit.setGeometry(QtCore.QRect(80, 140, 421, 40))
        self.browse_lineEdit.setObjectName("browse_lineEdit")
        self.upload_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.upload_pushButton.setGeometry(QtCore.QRect(80, 205, 120, 40))
        self.upload_pushButton.setObjectName("upload_pushButton")
        self.browse_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.browse_pushButton.setGeometry(QtCore.QRect(510, 140, 120, 40))
        self.browse_pushButton.setObjectName("browse_pushButton")
        self.image_upload_error_label = QtWidgets.QLabel(self.centralwidget)
        self.image_upload_error_label.setGeometry(QtCore.QRect(80, 182, 321, 16))
        self.image_upload_error_label.setStyleSheet("color: rgb(227, 0, 4)")
        self.image_upload_error_label.setObjectName("image_upload_error_label")

        self.gender_listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.gender_listWidget.setGeometry(QtCore.QRect(80, 370, 256, 192))
        self.gender_listWidget.setObjectName("gender_listWidget")
        self.gender_listWidget.setSelectionMode(QAbstractItemView.MultiSelection)
        item = QtWidgets.QListWidgetItem()
        self.gender_listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.gender_listWidget.addItem(item)

        self.which_audience_label = QtWidgets.QLabel(self.centralwidget)
        self.which_audience_label.setGeometry(QtCore.QRect(80, 280, 481, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.which_audience_label.setFont(font)
        self.which_audience_label.setObjectName("which_audience_label")

        self.gender_label = QtWidgets.QLabel(self.centralwidget)
        self.gender_label.setGeometry(QtCore.QRect(80, 325, 251, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)

        self.gender_label.setFont(font)
        self.gender_label.setObjectName("gender_label")
        self.gender_information_label = QtWidgets.QLabel(self.centralwidget)
        self.gender_information_label.setGeometry(QtCore.QRect(80, 350, 321, 16))
        self.gender_information_label.setObjectName("gender_information_label")

        self.gender_error_label = QtWidgets.QLabel(self.centralwidget)
        self.gender_error_label.setGeometry(QtCore.QRect(80, 570, 321, 16))
        self.gender_error_label.setStyleSheet("color: rgb(227, 0, 4)")
        self.gender_error_label.setObjectName("gender_error_label")
        self.age_label = QtWidgets.QLabel(self.centralwidget)
        self.age_label.setGeometry(QtCore.QRect(430, 325, 251, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.age_label.setFont(font)
        self.age_label.setObjectName("age_label")
        self.age_information_label = QtWidgets.QLabel(self.centralwidget)
        self.age_information_label.setGeometry(QtCore.QRect(430, 350, 321, 16))
        self.age_information_label.setObjectName("age_information_label")
        self.age_listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.age_listWidget.setGeometry(QtCore.QRect(430, 370, 256, 192))
        self.age_listWidget.setObjectName("age_listWidget")
        self.age_listWidget.setSelectionMode(QAbstractItemView.MultiSelection)
        item = QtWidgets.QListWidgetItem()
        self.age_listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.age_listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.age_listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.age_listWidget.addItem(item)
        self.age_error_label = QtWidgets.QLabel(self.centralwidget)
        self.age_error_label.setGeometry(QtCore.QRect(430, 570, 321, 16))
        self.age_error_label.setStyleSheet("color: rgb(227, 0, 4)")
        self.age_error_label.setObjectName("age_error_label")
        self.emotion_listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.emotion_listWidget.setGeometry(QtCore.QRect(790, 370, 256, 192))
        self.emotion_listWidget.setObjectName("emotion_listWidget")
        self.emotion_listWidget.setSelectionMode(QAbstractItemView.MultiSelection)
        item = QtWidgets.QListWidgetItem()
        self.emotion_listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.emotion_listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.emotion_listWidget.addItem(item)
        self.emotion_error_label = QtWidgets.QLabel(self.centralwidget)
        self.emotion_error_label.setGeometry(QtCore.QRect(790, 570, 321, 16))
        self.emotion_error_label.setStyleSheet("color: rgb(227, 0, 4)")
        self.emotion_error_label.setObjectName("emotion_error_label")
        self.emotion_label = QtWidgets.QLabel(self.centralwidget)
        self.emotion_label.setGeometry(QtCore.QRect(790, 325, 251, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.emotion_label.setFont(font)
        self.emotion_label.setObjectName("emotion_label")
        self.emotion_information_label = QtWidgets.QLabel(self.centralwidget)
        self.emotion_information_label.setGeometry(QtCore.QRect(790, 350, 321, 16))
        self.emotion_information_label.setObjectName("emotion_information_label")
        self.image_prewiev_imageLabel = QtWidgets.QLabel(self.centralwidget)
        self.image_prewiev_imageLabel.setGeometry(QtCore.QRect(700, 60, 350, 200))
        self.image_prewiev_imageLabel.setMaximumSize(QtCore.QSize(350, 200))
        self.image_prewiev_imageLabel.setText("")
        self.image_prewiev_imageLabel.setObjectName("image_prewiev_imageLabel")
        self.image_prewiev_textLabel = QtWidgets.QLabel(self.centralwidget)
        self.image_prewiev_textLabel.setGeometry(QtCore.QRect(700, 30, 381, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.image_prewiev_textLabel.setFont(font)
        self.image_prewiev_textLabel.setObjectName("image_prewiev_textLabel")

        """self.all_upload_ads_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.all_upload_ads_pushButton.setGeometry(QtCore.QRect(80, 660, 160, 28))
        self.all_upload_ads_pushButton.setObjectName("all_upload_ads_pushButton")"""

        self.save_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.save_pushButton.setGeometry(QtCore.QRect(80, 600, 160, 40))
        self.save_pushButton.setObjectName("save_pushButton")

        self.new_image_upload_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.new_image_upload_pushButton.setGeometry(QtCore.QRect(250, 600, 160, 28))
        self.new_image_upload_pushButton.setObjectName("new_image_upload_pushButton")
        self.new_image_upload_pushButton.hide()


        self.save_is_ready_label = QtWidgets.QLabel(self.centralwidget)
        self.save_is_ready_label.setGeometry(QtCore.QRect(80, 630, 160, 28))
        self.save_is_ready_label.setObjectName("save_is_ready_label")
        self.save_is_ready_label.setAlignment(QtCore.Qt.AlignCenter)

        """Dialog.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Dialog)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 26))
        self.menubar.setObjectName("menubar")
        Dialog.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Dialog)
        self.statusbar.setObjectName("statusbar")
        Dialog.setStatusBar(self.statusbar)"""

        # click button
        self.browse_pushButton.clicked.connect(self.click_browse_button)
        self.upload_pushButton.clicked.connect(self.click_upload_button)
        self.save_pushButton.clicked.connect(self.click_save_button)
        self.new_image_upload_pushButton.clicked.connect(self.click_new_image_upload_button)

        # set texts
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Hirdetés feltöltése"))
        self.ad_name_label.setText(_translate("Dialog", "Kérem nevezze el a hirdetést!"))
        self.image_upload_label.setText(_translate("Dialog", "Kérem válassza ki a feltölteni kívánt képet!"))
        self.upload_pushButton.setText(_translate("Dialog", "Feltöltés"))
        self.browse_pushButton.setText(_translate("Dialog", "Tallózás"))
        self.image_upload_error_label.setText(_translate("Dialog", ""))
        __sortingEnabled = self.gender_listWidget.isSortingEnabled()
        self.gender_listWidget.setSortingEnabled(False)
        item = self.gender_listWidget.item(0)
        item.setText(_translate("Dialog", "Nő"))
        item = self.gender_listWidget.item(1)
        item.setText(_translate("Dialog", "Férfi"))
        self.gender_listWidget.setSortingEnabled(__sortingEnabled)
        self.which_audience_label.setText(
            _translate("Dialog", "Kérem adja meg, milyen célcsoportnak kíván hirdetni!"))
        self.gender_label.setText(_translate("Dialog", "Nem"))
        self.gender_information_label.setText(_translate("Dialog", "(Kérem válasszon egyet vagy többet)"))
        self.gender_error_label.setText(_translate("Dialog", ""))
        self.age_label.setText(_translate("Dialog", "Korcsoport"))
        self.age_information_label.setText(_translate("Dialog", "(Kérem válasszon egyet vagy többet)"))
        __sortingEnabled = self.age_listWidget.isSortingEnabled()
        self.age_listWidget.setSortingEnabled(False)
        item = self.age_listWidget.item(0)
        item.setText(_translate("Dialog", "Gyerek (18 év alatti)"))
        item = self.age_listWidget.item(1)
        item.setText(_translate("Dialog", "Fiatal felnőtt"))
        item = self.age_listWidget.item(2)
        item.setText(_translate("Dialog", "Középkorú"))
        item = self.age_listWidget.item(3)
        item.setText(_translate("Dialog", "Idős"))
        self.age_listWidget.setSortingEnabled(__sortingEnabled)
        self.age_error_label.setText(_translate("Dialog", ""))
        __sortingEnabled = self.emotion_listWidget.isSortingEnabled()
        self.emotion_listWidget.setSortingEnabled(False)
        item = self.emotion_listWidget.item(0)
        item.setText(_translate("Dialog", "Boldog"))
        item = self.emotion_listWidget.item(1)
        item.setText(_translate("Dialog", "Semleges"))
        item = self.emotion_listWidget.item(2)
        item.setText(_translate("Dialog", "Szomorú"))
        self.emotion_listWidget.setSortingEnabled(__sortingEnabled)
        self.emotion_error_label.setText(_translate("Dialog", ""))
        self.emotion_label.setText(_translate("Dialog", "Érzelmi állapot"))
        self.emotion_information_label.setText(_translate("Dialog", "(Kérem válasszon egyet vagy többet)"))
        self.image_prewiev_textLabel.setText(_translate("Dialog", "A kép előnézete"))
        self.save_pushButton.setText(_translate("Dialog", "Mentés"))
        self.save_is_ready_label.setText(_translate("Dialog", ""))
        self.new_image_upload_pushButton.setText(_translate("Dialog", "Új kép feltöltése"))
        #self.all_upload_ads_pushButton.setText(_translate("Dialog", "Feltöltött hirdetések"))

    def click_browse_button(self):
        options = QFileDialog.Options()
        file_url, _ = QFileDialog.getOpenFileName(None, "Fájl Tallózása", "", "Minden fájl (*.*)", options=options)

        if file_url:
            self.browse_lineEdit.setText(file_url)

    def click_upload_button(self):
        file_path = self.browse_lineEdit.text()
        ads_img_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)

        if ads_img_bgr is None:
            self.image_upload_error_label.setText("A fájl a formátuma nem megfelelő.")
            self.browse_lineEdit.setText("Kérlek válassz ki egy létező képet!")
            self.image_prewiev_imageLabel.clear()
            return

        else:
            self.image_upload_error_label.setText("")
            ads_img = cv2.cvtColor(ads_img_bgr, cv2.COLOR_BGR2RGB)
            height, width, channel = ads_img.shape

            if width > 350:
                new_width = 350
                new_height = int((new_width / width) * height)

                if new_height > 200:
                    new_new_height = 200
                    new_new_width = int((new_new_height / new_height) * new_width)
                    new_height = new_new_height
                    new_width = new_new_width

                ads_img = cv2.resize(ads_img, (new_width, new_height))
                height = new_height
                width = new_width

            bytes_per_line = 3 * width
            q_ads_img = QImage(ads_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap_img = QPixmap.fromImage(q_ads_img)
            self.image_prewiev_imageLabel.setPixmap(pixmap_img)

    def click_save_button(self):

        ad_name = self.ad_name_lineEdit.text()

        if ad_name == "":
            self.ad_name_error_label.setText("Kérlek adj meg egy nevet!")
            return
        else:
            self.ad_name_error_label.setText("")


        file_path = self.browse_lineEdit.text()
        ads_img_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)

        if ads_img_bgr is None:
            self.image_upload_error_label.setText("A fájl a formátuma nem megfelelő.")
            self.browse_lineEdit.setText("Kérlek válassz ki egy létező képet!")
            self.image_prewiev_imageLabel.clear()
            return

        selected_gender = self.gender_listWidget.selectedItems()

        if not selected_gender:
            self.gender_error_label.setText("Kérem válasszon ki legalább egyet!")
        else:
            self.gender_error_label.setText("")

        selected_ages = self.age_listWidget.selectedItems()

        if not selected_ages:
            self.age_error_label.setText("Kérem válasszon ki legalább egyet!")
        else:
            self.age_error_label.setText("")

        selected_emotion = self.emotion_listWidget.selectedItems()

        if not selected_emotion:
            self.emotion_error_label.setText("Kérem válasszon ki legalább egyet!")
        else:
            self.emotion_error_label.setText("")

        female = 0
        male = 0

        for i in selected_gender:
            if i.text() == "Nő":
                female = 1
            else:
                male = 1

        child = 0
        young_adult = 0
        middle_aged = 0
        elderly = 0

        for i in selected_ages:
            if i.text() == "Gyerek (18 év alatti)":
                child = 1
            elif i.text() == "Fiatal felnőtt":
                young_adult = 1
            elif i.text() == "Középkorú":
                middle_aged = 1
            else:
                elderly = 1

        happy = 0
        neutral = 0
        sad = 0

        for i in selected_emotion:
            if i.text() == "Boldog":
                happy = 1
            elif i.text() == "Semleges":
                neutral = 1
            else:
                sad = 1

        date_now = datetime.now()
        upload_date = date_now.strftime("%Y-%m-%d %H:%M:%S")


        conn = sqlite3.connect('picturesdatabase.db')
        cursor = conn.cursor()

        cursor.execute('''INSERT INTO images 
                       (filename, ad_name, upload_date, female, male, child, young_adult, middle_aged, elderly, happy, neutral, sad) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (file_path, ad_name, upload_date, female, male, child, young_adult, middle_aged, elderly, happy, neutral, sad))

        image_id = cursor.lastrowid
        new_file_path = "pictures/" + str(image_id) + ".jpg"
        cv2.imwrite(new_file_path,ads_img_bgr)
        cursor.execute(''' UPDATE images SET filename = ?
                        WHERE id = ?''',(new_file_path,image_id))


        conn.commit()  # Az adatok mentése
        conn.close()  # Az adatbázis bezárása

        self.save_is_ready_label.setText("Sikeres feltöltés")
        self.new_image_upload_pushButton.show()

        self.ad_name_lineEdit.setEnabled(False)
        self.browse_pushButton.setEnabled(False)
        self.upload_pushButton.setEnabled(False)
        self.save_pushButton.setEnabled(False)
        self.browse_lineEdit.setEnabled(False)


    def click_new_image_upload_button(self):

        self.browse_lineEdit.setText("")
        self.save_is_ready_label.setText("")
        self.ad_name_lineEdit.setText("")
        self.age_listWidget.clearSelection()
        self.gender_listWidget.clearSelection()
        self.emotion_listWidget.clearSelection()
        self.image_prewiev_imageLabel.clear()
        self.new_image_upload_pushButton.hide()
        self.ad_name_lineEdit.setEnabled(True)
        self.browse_pushButton.setEnabled(True)
        self.upload_pushButton.setEnabled(True)
        self.save_pushButton.setEnabled(True)
        self.browse_lineEdit.setEnabled(True)



if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Upload_Image_Window()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
